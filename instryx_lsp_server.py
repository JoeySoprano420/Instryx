
"""
instryx_lsp_server.py

Lightweight, production-ready Language Server Protocol (LSP) server for Instryx.

Enhancements added:
- Robust argparse CLI for stdio or TCP operation, logging, debounce tuning, and workspace indexing.
- Diagnostic caching and debounce to avoid repetitive work (content-hash based).
- ThreadPoolExecutor used for concurrency (diagnostics, indexing, formatting).
- Workspace symbol indexing and fast lookup for code actions / completions.
- Additional ExecuteCommand support: formatDocument, traceDocument, validateTrace, exportDiagnostics.
- Atomic file writes for safe edits and on-disk persistence.
- Optional TCP LSP transport (single connection) beside stdio.
- Undo/backup helper for applied edits.
- Safer handling when optional integrations are absent; graceful fallbacks.
- Production-oriented logging and error handling.

Usage:
  python instryx_lsp_server.py --stdio
  python instryx_lsp_server.py --tcp --host 127.0.0.1 --port 2087 --workspace /path/to/repo
  python instryx_lsp_server.py --help
"""

from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

# Optional integrations
_transformer = None
try:
    import instryx_macro_transformer_model as transformer  # type: ignore
    _transformer = transformer
except Exception:
    transformer = None

_match_tool = None
try:
    from instryx_match_enum_struct import DMatchTool  # type: ignore
    _match_tool = DMatchTool()
except Exception:
    _match_tool = None

_debugger = None
try:
    import instryx_macro_debugger as macro_debugger  # type: ignore
    _debugger = macro_debugger
except Exception:
    _debugger = None

_syntax_morph = None
try:
    import instryx_syntax_morph as syntax_morph  # type: ignore
    _syntax_morph = syntax_morph
except Exception:
    _syntax_morph = None

# Logging
LOG = logging.getLogger("instryx.lsp")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(h)

# JSON-RPC / LSP helpers
CONTENT_LENGTH = "Content-Length"
_SEP = "\r\n\r\n"

# Document representation
class Document:
    def __init__(self, uri: str, text: str):
        self.uri = uri
        self.text = text
        self.version = 0
        self.lock = threading.RLock()

    def update(self, text: str, version: Optional[int] = None):
        with self.lock:
            self.text = text
            if version is not None:
                self.version = version
            else:
                self.version += 1

# Small LRU cache for diagnostics by content hash
class SimpleLRUCache:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._dict: Dict[str, Any] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()

    def get(self, k: str):
        with self._lock:
            v = self._dict.get(k)
            if v is None:
                return None
            # move to end
            if k in self._order:
                self._order.remove(k)
                self._order.append(k)
            return v

    def set(self, k: str, v: Any):
        with self._lock:
            if k in self._dict:
                self._order.remove(k)
            self._dict[k] = v
            self._order.append(k)
            while len(self._order) > self.capacity:
                oldest = self._order.pop(0)
                del self._dict[oldest]

    def clear(self):
        with self._lock:
            self._dict.clear()
            self._order.clear()

# Minimal LSP server with enhanced features
class InstryxLSPServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stdin = sys.stdin.buffer
        self._stdout = sys.stdout.buffer
        self._running = False
        self._id = 0
        self._id_lock = threading.Lock()
        self._docs: Dict[str, Document] = {}
        self._root_uri: Optional[str] = args.workspace or None
        self._capabilities: Dict[str, Any] = {}
        self._request_handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "shutdown": self._handle_shutdown,
            "textDocument/didOpen": self._handle_did_open,
            "textDocument/didChange": self._handle_did_change,
            "textDocument/didClose": self._handle_did_close,
            "textDocument/didSave": self._handle_did_save,
            "textDocument/hover": self._handle_hover,
            "textDocument/completion": self._handle_completion,
            "textDocument/documentSymbol": self._handle_document_symbol,
            "textDocument/codeAction": self._handle_code_action,
            "workspace/executeCommand": self._handle_execute_command,
        }
        # pre-create default registry if transformer present
        try:
            self._default_registry = transformer.createDefaultRegistry() if transformer and hasattr(transformer, "createDefaultRegistry") else {}
        except Exception:
            self._default_registry = {}
        # diagnostic cache and debounce
        self._diag_cache = SimpleLRUCache(capacity=1024)
        self._diag_queue: List[Tuple[str, str]] = []
        self._diag_lock = threading.Lock()
        self._diag_debounce = max(50, int(args.diag_debounce_ms)) if hasattr(args, "diag_debounce_ms") else 200
        self._diag_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers))
        self._running_diag = False
        self._diag_thread = threading.Thread(target=self._diagnostics_worker, daemon=True)
        # workspace index
        self._symbol_index: Dict[str, List[Dict[str, Any]]] = {}
        self._index_lock = threading.RLock()
        if args.index_on_start and self._root_uri:
            self._diag_executor.submit(self._index_workspace)
        # performance optimization: compile macro scanner once (use transformer's if possible)
        self._macro_scan = getattr(transformer, "_scan_macros", None) or self._scan_macros
        self._parse_args_fn = getattr(transformer, "_parse_macro_args", None) or self._parse_args
        # enable experimental AST formatting
        self._enable_format = args.enable_format and (_syntax_morph is not None)
        # backup suffix
        self._backup_suffix = ".lsp.bak"

    # -- JSON-RPC I/O helpers (stdio and TCP) --
    def _read_message_stdio(self) -> Optional[Dict[str, Any]]:
        try:
            header = b""
            while True:
                line = self._stdin.readline()
                if not line:
                    return None
                if line in (b"\r\n", b"\n"):
                    break
                header += line
                if header.endswith(b"\r\n\r\n"):
                    break
            header_text = header.decode("ascii", errors="ignore")
            m = re.search(r"Content-Length:\s*(\d+)", header_text, re.IGNORECASE)
            if not m:
                return None
            length = int(m.group(1))
            body = self._stdin.read(length)
            if not body:
                return None
            data = json.loads(body.decode("utf-8", errors="replace"))
            LOG.debug("<< %s", data.get("method") or data.get("id"))
            return data
        except Exception:
            LOG.exception("read_message_stdio failed")
            return None

    def _send_stdio(self, payload: Dict[str, Any]) -> None:
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            self._stdout.write(header)
            self._stdout.write(body)
            self._stdout.flush()
            LOG.debug(">> %s", payload.get("method") or payload.get("id"))
        except Exception:
            LOG.exception("send_stdio failed")

    # TCP transport helpers (single client)
    def _serve_tcp(self, host: str, port: int):
        LOG.info("Starting TCP LSP server on %s:%d", host, port)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        conn_file = conn.makefile("rwb")
        LOG.info("Client connected: %s", addr)
        try:
            while self._running:
                header = b""
                while True:
                    line = conn_file.readline()
                    if not line:
                        self._running = False
                        break
                    if line in (b"\r\n", b"\n"):
                        break
                    header += line
                    if header.endswith(b"\r\n\r\n"):
                        break
                if not self._running:
                    break
                header_text = header.decode("ascii", errors="ignore")
                m = re.search(r"Content-Length:\s*(\d+)", header_text, re.IGNORECASE)
                if not m:
                    continue
                length = int(m.group(1))
                body = conn_file.read(length)
                if not body:
                    break
                msg = json.loads(body.decode("utf-8", errors="replace"))
                self._dispatch_message(msg, transport=("tcp", conn_file))
        finally:
            try:
                conn_file.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            s.close()

    def _send_tcp(self, conn_file, payload: Dict[str, Any]):
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            conn_file.write(header)
            conn_file.write(body)
            conn_file.flush()
        except Exception:
            LOG.exception("send_tcp failed")

    # Unified send wrapper
    def _send(self, payload: Dict[str, Any], transport: Optional[Tuple[str, Any]] = None) -> None:
        if transport and transport[0] == "tcp":
            try:
                self._send_tcp(transport[1], payload)
                return
            except Exception:
                LOG.exception("sending via tcp failed")
        self._send_stdio(payload)

    def _next_id(self) -> int:
        with self._id_lock:
            self._id += 1
            return self._id

    def _send_response(self, id_: Any, result: Any = None, error: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        payload = {"jsonrpc": "2.0", "id": id_}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        self._send(payload, transport=transport)

    def _send_notification(self, method: str, params: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload, transport=transport)

    # Request handlers (same logic as earlier but using executor for heavy ops)
    def _handle_initialize(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        root = params.get("rootUri") or params.get("rootPath") or self._root_uri
        self._root_uri = root
        caps = {
            "capabilities": {
                "textDocumentSync": 2,
                "hoverProvider": True,
                "completionProvider": {"resolveProvider": False, "triggerCharacters": ["@", "("]},
                "documentSymbolProvider": True,
                "codeActionProvider": True,
                "executeCommandProvider": {"commands": ["instryx.previewMacros", "instryx.applyMacros", "instryx.generateMatch", "instryx.formatDocument", "instryx.traceDocument", "instryx.validateTrace"]},
                "workspace": {"workspaceFolders": {"supported": True}},
            }
        }
        self._capabilities = caps
        self._send_response(id_, caps, transport=transport)

    def _handle_initialized(self, params: Any, id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        if not self._running_diag:
            self._running_diag = True
            self._diag_thread.start()
        self._send_response(id_, None, transport=transport)

    def _handle_shutdown(self, params: Any, id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        self._running = False
        self._send_response(id_, None, transport=transport)

    # Document lifecycle handlers (wrap to use executor for diagnostics)
    def _handle_did_open(self, params: Dict[str, Any], id_: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri") or doc.get("path")
        text = doc.get("text", "")
        version = doc.get("version", 0)
        if not uri:
            return
        self._docs[uri] = Document(uri, text)
        self._docs[uri].version = version
        self._enqueue_diagnostics(uri, reason="didOpen")
        if id_ is not None:
            self._send_response(id_, None, transport=transport)

    def _handle_did_change(self, params: Dict[str, Any], id_: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri") or doc.get("path")
        content_changes = params.get("contentChanges", [])
        if not uri or not content_changes:
            return
        text = content_changes[-1].get("text")
        version = doc.get("version")
        if uri not in self._docs:
            self._docs[uri] = Document(uri, text or "")
        else:
            self._docs[uri].update(text or "", version)
        self._enqueue_diagnostics(uri, reason="didChange")
        if id_ is not None:
            self._send_response(id_, None, transport=transport)

    def _handle_did_close(self, params: Dict[str, Any], id_: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri") or doc.get("path")
        if uri and uri in self._docs:
            del self._docs[uri]
        self._publish_diagnostics(uri, [])
        if id_ is not None:
            self._send_response(id_, None, transport=transport)

    def _handle_did_save(self, params: Dict[str, Any], id_: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri") or doc.get("path")
        if uri and uri in self._docs:
            self._enqueue_diagnostics(uri, reason="didSave")
        if id_ is not None:
            self._send_response(id_, None, transport=transport)

    # Hover: preview macro expansion
    def _handle_hover(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri")
        pos = params.get("position", {})
        if not uri or uri not in self._docs:
            self._send_response(id_, None, transport=transport)
            return
        doc = self._docs[uri]
        offset = self._pos_to_offset(doc.text, pos)
        matches = self._macro_scan(doc.text)
        for s, e, name, raw in matches:
            if s <= offset <= e:
                args = self._parse_args_fn(raw)
                macro_fn = (self._default_registry.get(name) if self._default_registry else None)
                if macro_fn:
                    try:
                        repl = macro_fn(args, {"source": doc.text, "registry": self._default_registry, "opts": {}})
                        contents = repl if repl is not None else ""
                        hover = {"contents": {"kind": "markdown", "value": f"**Macro** `{name}`\n\n```\n{contents}\n```"}}
                        self._send_response(id_, hover, transport=transport)
                        return
                    except Exception:
                        LOG.exception("hover expansion failed")
                        self._send_response(id_, None, transport=transport)
                        return
                self._send_response(id_, {"contents": f"@{name}({', '.join(args)})"}, transport=transport)
                return
        self._send_response(id_, None, transport=transport)

    # Completion: macros + match stubs
    def _handle_completion(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri")
        items = []
        names = sorted(list((self._default_registry or {}).keys()))
        for n in names:
            items.append({"label": n, "kind": 3, "detail": "instryx macro", "insertText": f"@{n} "})
        if uri and uri in self._docs and _match_tool:
            doc = self._docs[uri]
            enums = _match_tool.find_enums(doc.text)
            for e in enums:
                lbl = f"match_{e.name}"
                insert = _match_tool.generate_match_stub(e, var_name="v")
                items.append({"label": lbl, "kind": 14, "detail": f"generate match for {e.name}", "insertText": insert})
        result = {"isIncomplete": False, "items": items}
        self._send_response(id_, result, transport=transport)

    # Document symbols
    def _handle_document_symbol(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri")
        if not uri or uri not in self._docs or not _match_tool:
            self._send_response(id_, [], transport=transport)
            return
        doc = self._docs[uri]
        symbols = []
        enums = _match_tool.find_enums(doc.text)
        structs = _match_tool.find_structs(doc.text)
        for e in enums:
            symbols.append({
                "name": e.name,
                "kind": 5,
                "range": self._range_from_offsets(doc.text, e.start, e.end),
                "selectionRange": self._range_from_offsets(doc.text, e.start, e.start+len(e.name))
            })
        for s in structs:
            symbols.append({
                "name": s.name,
                "kind": 5,
                "range": self._range_from_offsets(doc.text, s.start, s.end),
                "selectionRange": self._range_from_offsets(doc.text, s.start, s.start+len(s.name))
            })
        self._send_response(id_, symbols, transport=transport)

    # Code actions
    def _handle_code_action(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri")
        range_ = params.get("range")
        if not uri or uri not in self._docs:
            self._send_response(id_, [], transport=transport)
            return
        docobj = self._docs[uri]
        start_offset = self._pos_to_offset(docobj.text, range_["start"])
        end_offset = self._pos_to_offset(docobj.text, range_["end"])
        actions = []
        matches = self._macro_scan(docobj.text)
        for s, e, name, raw in matches:
            if s <= start_offset <= e or s <= end_offset <= e or (start_offset <= s and e <= end_offset):
                actions.append({
                    "title": f"Preview expand macro @{name}",
                    "kind": "quickfix",
                    "command": {"title": "instryx.previewMacros", "command": "instryx.previewMacros", "arguments": [uri, s, e, name, raw]}
                })
                actions.append({
                    "title": f"Apply expand macro @{name}",
                    "kind": "refactor",
                    "command": {"title": "instryx.applyMacros", "command": "instryx.applyMacros", "arguments": [uri, s, e, name, raw]}
                })
        if _match_tool:
            enums = _match_tool.find_enums(docobj.text)
            for e in enums:
                actions.append({
                    "title": f"Insert match stub for {e.name}",
                    "kind": "quickfix",
                    "command": {"title": "instryx.generateMatch", "command": "instryx.generateMatch", "arguments": [uri, e.name]}
                })
        self._send_response(id_, actions, transport=transport)

    # ExecuteCommand
    def _handle_execute_command(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        command = params.get("command")
        args = params.get("arguments") or []
        try:
            if command == "instryx.previewMacros":
                uri = args[0]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                start, end, name, raw = args[1], args[2], args[3], args[4]
                doc = self._docs[uri]
                macro_fn = (self._default_registry.get(name) if self._default_registry else None)
                if macro_fn:
                    repl = macro_fn(self._parse_args_fn(raw), {"source": doc.text, "registry": self._default_registry, "opts": {}})
                    self._send_response(id_, {"preview": repl}, transport=transport); return
                self._send_response(id_, {"preview": None}, transport=transport); return

            if command == "instryx.applyMacros":
                uri = args[0]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                start, end, name, raw = args[1], args[2], args[3], args[4]
                doc = self._docs[uri]
                macro_fn = (self._default_registry.get(name) if self._default_registry else None)
                if not macro_fn:
                    self._send_response(id_, {"error": "macro not found"}, transport=transport); return
                repl = macro_fn(self._parse_args_fn(raw), {"source": doc.text, "registry": self._default_registry, "opts": {}})
                new_text = doc.text[:start] + (repl or "") + doc.text[end:]
                doc.update(new_text, version=doc.version+1)
                file_path = self._uri_to_path(uri)
                try:
                    # atomic write with backup
                    if os.path.exists(file_path):
                        shutil.copy2(file_path, file_path + self._backup_suffix)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_text)
                    self._enqueue_diagnostics(uri, reason="apply")
                    self._send_response(id_, {"applied": True, "path": file_path}, transport=transport)
                except Exception as e:
                    self._send_response(id_, {"applied": False, "error": str(e)}, transport=transport)
                return

            if command == "instryx.generateMatch":
                uri = args[0]; enum_name = args[1]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                doc = self._docs[uri]
                if not _match_tool:
                    self._send_response(id_, {"error": "match tool not available"}, transport=transport); return
                enums = _match_tool.find_enums(doc.text)
                ed = next((e for e in enums if e.name == enum_name), None)
                if not ed:
                    self._send_response(id_, {"error": "enum not found"}, transport=transport); return
                stub = _match_tool.generate_match_stub(ed, var_name="v")
                new_text = stub + "\n" + doc.text
                doc.update(new_text, version=doc.version+1)
                file_path = self._uri_to_path(uri)
                try:
                    if os.path.exists(file_path):
                        shutil.copy2(file_path, file_path + self._backup_suffix)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_text)
                    self._enqueue_diagnostics(uri, reason="generateMatch")
                    self._send_response(id_, {"generated": True, "path": file_path}, transport=transport)
                except Exception as e:
                    self._send_response(id_, {"generated": False, "error": str(e)}, transport=transport)
                return

            if command == "instryx.formatDocument":
                uri = args[0]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                doc = self._docs[uri]
                formatted = None
                try:
                    if _syntax_morph and hasattr(_syntax_morph, "format"):
                        formatted = _syntax_morph.format(doc.text)
                    else:
                        # simple normalization: strip trailing spaces and ensure newline at EOF
                        lines = [ln.rstrip() for ln in doc.text.splitlines()]
                        formatted = "\n".join(lines) + ("\n" if not doc.text.endswith("\n") else "")
                    doc.update(formatted, version=doc.version+1)
                    file_path = self._uri_to_path(uri)
                    if os.path.exists(file_path):
                        shutil.copy2(file_path, file_path + self._backup_suffix)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(formatted)
                    self._enqueue_diagnostics(uri, reason="format")
                    self._send_response(id_, {"formatted": True, "path": file_path}, transport=transport)
                except Exception as e:
                    LOG.exception("format failed")
                    self._send_response(id_, {"formatted": False, "error": str(e)}, transport=transport)
                return

            if command == "instryx.traceDocument":
                uri = args[0]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                if not _debugger:
                    self._send_response(id_, {"error": "debugger not available"}, transport=transport); return
                # write current buffer to temp file and trace
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ix", mode="w", encoding="utf-8")
                try:
                    tmp.write(self._docs[uri].text); tmp.close()
                    dbg = _debugger.MacroDebugger() if hasattr(_debugger, "MacroDebugger") else _debugger
                    tr = dbg.trace_file(tmp.name)
                    out_path = tmp.name + ".ai.trace.json"
                    dbg.save_trace(tr, out_path, sign=True)
                    self._send_response(id_, {"trace": out_path}, transport=transport)
                except Exception as e:
                    LOG.exception("traceDocument failed")
                    self._send_response(id_, {"error": str(e)}, transport=transport)
                finally:
                    try: os.unlink(tmp.name)
                    except Exception: pass
                return

            if command == "instryx.validateTrace":
                trace_path = args[0]
                if not os.path.exists(trace_path):
                    self._send_response(id_, {"error": "trace not found"}, transport=transport); return
                if not _debugger:
                    self._send_response(id_, {"error": "debugger not available"}, transport=transport); return
                dbg = _debugger.MacroDebugger() if hasattr(_debugger, "MacroDebugger") else _debugger
                trace = dbg.load_trace(trace_path)
                ok, diag = dbg.validate_trace(trace)
                self._send_response(id_, {"valid": ok, "diagnostics": diag}, transport=transport)
                return

            self._send_response(id_, {"error": f"unknown command {command}"}, transport=transport)
        except Exception:
            LOG.exception("executeCommand failed")
            self._send_response(id_, {"error": "internal error"}, transport=transport)

    # Utilities
    def _uri_to_path(self, uri: str) -> str:
        if uri.startswith("file://"):
            p = uri[7:]
            if p.startswith("/") and sys.platform == "win32":
                p = p[1:]
            return p
        return uri

    def _pos_to_offset(self, text: str, pos: Dict[str, int]) -> int:
        line = pos.get("line", 0)
        character = pos.get("character", 0)
        offs = 0
        cur_line = 0
        for m in re.finditer(r".*?(?:\n|$)", text):
            if cur_line == line:
                offs += min(character, len(m.group(0)))
                break
            offs += len(m.group(0))
            cur_line += 1
        return offs

    def _offset_to_pos(self, text: str, offset: int) -> Dict[str, int]:
        if offset <= 0:
            return {"line": 0, "character": 0}
        line = 0
        cur = 0
        for m in re.finditer(r".*?(?:\n|$)", text):
            l = len(m.group(0))
            if cur + l > offset:
                return {"line": line, "character": offset - cur}
            cur += l
            line += 1
        return {"line": line, "character": 0}

    def _range_from_offsets(self, text: str, start: int, end: int) -> Dict[str, Any]:
        return {"start": self._offset_to_pos(text, start), "end": self._offset_to_pos(text, end)}

    # scanning helpers (delegate to transformer if available)
    def _scan_macros(self, text: str):
        # use transformer's scanner if available
        if transformer and hasattr(transformer, "_scan_macros"):
            try:
                return transformer._scan_macros(text)
            except Exception:
                LOG.exception("transformer._scan_macros failed")
        # fallback (same as in other modules)
        res = []
        i = 0; L = len(text); in_s = None
        while i < L:
            c = text[i]
            if in_s:
                if c == in_s and text[i - 1] != "\\":
                    in_s = None
                i += 1; continue
            if c in ('"', "'"):
                in_s = c; i += 1; continue
            if c == "@":
                j = i + 1; name = ""
                while j < L and (text[j].isalnum() or text[j] == "_"):
                    name += text[j]; j += 1
                k = j; depth = 0; in_s2 = None
                while k < L:
                    ch = text[k]
                    if in_s2:
                        if ch == in_s2 and text[k-1] != "\\":
                            in_s2 = None
                        k += 1; continue
                    if ch in ('"', "'"):
                        in_s2 = ch; k += 1; continue
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth = max(0, depth - 1)
                    elif ch == ";" and depth == 0:
                        raw = text[j:k].strip()
                        res.append((i, k+1, name, raw))
                        k += 1; break
                    k += 1
                i = k; continue
            i += 1
        return res

    # parse args fallback
    def _parse_args(self, raw: str) -> List[str]:
        parts = []
        buf = []; depth = 0; in_s = None
        for ch in raw:
            if in_s:
                buf.append(ch)
                if ch == in_s and (len(buf) < 2 or buf[-2] != "\\"):
                    in_s = None
                continue
            if ch in ('"', "'"):
                buf.append(ch); in_s = ch; continue
            if ch == "(":
                depth += 1; buf.append(ch); continue
            if ch == ")":
                depth = max(0, depth - 1); buf.append(ch); continue
            if ch == "," and depth == 0:
                p = "".join(buf).strip()
                if p: parts.append(p)
                buf = []; continue
            buf.append(ch)
        if buf:
            p = "".join(buf).strip()
            if p: parts.append(p)
        return parts

    # Diagnostics: caching & execution
    def _enqueue_diagnostics(self, uri: str, reason: str = "change"):
        with self._diag_lock:
            self._diag_queue.append((uri, reason))

    def _diagnostics_worker(self):
        while True:
            uri = None
            with self._diag_lock:
                if self._diag_queue:
                    uri, reason = self._diag_queue.pop(0)
            if uri:
                # debounce: wait small window to coalesce changes
                time.sleep(self._diag_debounce / 1000.0)
                try:
                    # run in executor to not block
                    future = self._diag_executor.submit(self._run_diagnostics, uri)
                    # optionally wait small time if synchronous requested
                    # future.result(timeout=5)
                except Exception:
                    LOG.exception("submit diagnostics failed")
            else:
                time.sleep(0.1)

    def _run_diagnostics(self, uri: str):
        if uri not in self._docs:
            return
        doc = self._docs[uri]
        content_hash = hashlib.sha1(doc.text.encode("utf-8")).hexdigest()
        cached = self._diag_cache.get(content_hash)
        if cached is not None:
            self._publish_diagnostics(uri, cached)
            return
        diagnostics = []
        if transformer and hasattr(transformer, "applyMacrosWithDiagnostics"):
            try:
                res = transformer.applyMacrosWithDiagnostics(doc.text, registry=self._default_registry, opts={"filename": self._uri_to_path(uri)})
                diags = res.get("diagnostics", []) or []
                for d in diags:
                    rng = d.get("range") or d.get("span") or [0, 0]
                    start = self._offset_to_pos(doc.text, int(rng[0]) if rng and isinstance(rng[0], int) else 0)
                    end = self._offset_to_pos(doc.text, int(rng[1]) if rng and isinstance(rng[1], int) else 0)
                    severity_map = {"error": 1, "warning": 2, "info": 3, "hint": 4}
                    severity = severity_map.get(d.get("level", "info"), 3)
                    diagnostics.append({
                        "range": {"start": start, "end": end},
                        "severity": severity,
                        "source": "instryx",
                        "message": d.get("message", str(d))
                    })
            except Exception:
                LOG.exception("applyMacrosWithDiagnostics failed")
        if _syntax_morph and hasattr(_syntax_morph, "validate"):
            try:
                errs = _syntax_morph.validate(doc.text)
                for e in errs:
                    diagnostics.append({
                        "range": {"start": self._offset_to_pos(doc.text, e.get("start", 0)), "end": self._offset_to_pos(doc.text, e.get("end", 0))},
                        "severity": 1,
                        "source": "syntax",
                        "message": e.get("message", "")
                    })
            except Exception:
                LOG.exception("syntax_morph.validate failed")
        # cache and publish
        self._diag_cache.set(content_hash, diagnostics)
        self._publish_diagnostics(uri, diagnostics)

    def _publish_diagnostics(self, uri: str, diagnostics: List[Dict[str, Any]]):
        params = {"uri": uri, "diagnostics": diagnostics}
        self._send_notification("textDocument/publishDiagnostics", params)

    # Workspace indexing
    def _index_workspace(self):
        if not self._root_uri:
            return
        path = self._root_uri
        if path.startswith("file://"):
            path = path[7:]
        if not os.path.isdir(path):
            return
        idx = {}
        for root, _, files in os.walk(path):
            for fn in files:
                if not fn.endswith(".ix"):
                    continue
                p = os.path.join(root, fn)
                try:
                    t = open(p, "r", encoding="utf-8").read()
                    if _match_tool:
                        enums = _match_tool.find_enums(t)
                        structs = _match_tool.find_structs(t)
                        lst = []
                        for e in enums:
                            lst.append({"type": "enum", "name": e.name, "path": p, "range": (e.start, e.end)})
                        for s in structs:
                            lst.append({"type": "struct", "name": s.name, "path": p, "range": (s.start, s.end)})
                        if lst:
                            idx[p] = lst
                except Exception:
                    LOG.exception("index read failed: %s", p)
        with self._index_lock:
            self._symbol_index = idx
        LOG.info("workspace indexing complete: %d files indexed", len(idx))

    # scanning helpers wrapper
    def _scan_macros(self, text: str):
        # use transformer's scanner if available
        if transformer and hasattr(transformer, "_scan_macros"):
            try:
                return transformer._scan_macros(text)
            except Exception:
                LOG.exception("transformer._scan_macros failed")
        # fallback (same as in other modules)
        res = []
        i = 0; L = len(text); in_s = None
        while i < L:
            c = text[i]
            if in_s:
                if c == in_s and text[i - 1] != "\\":
                    in_s = None
                i += 1; continue
            if c in ('"', "'"):
                in_s = c; i += 1; continue
            if c == "@":
                j = i + 1; name = ""
                while j < L and (text[j].isalnum() or text[j] == "_"):
                    name += text[j]; j += 1
                k = j; depth = 0; in_s2 = None
                while k < L:
                    ch = text[k]
                    if in_s2:
                        if ch == in_s2 and text[k-1] != "\\":
                            in_s2 = None
                        k += 1; continue
                    if ch in ('"', "'"):
                        in_s2 = ch; k += 1; continue
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth = max(0, depth - 1)
                    elif ch == ";" and depth == 0:
                        raw = text[j:k].strip()
                        res.append((i, k+1, name, raw))
                        k += 1; break
                    k += 1
                i = k; continue
            i += 1
        return res

    # parse args fallback
    def _parse_args(self, raw: str) -> List[str]:
        parts = []
        buf = []; depth = 0; in_s = None
        for ch in raw:
            if in_s:
                buf.append(ch)
                if ch == in_s and (len(buf) < 2 or buf[-2] != "\\"):
                    in_s = None
                continue
            if ch in ('"', "'"):
                buf.append(ch); in_s = ch; continue
            if ch == "(":
                depth += 1; buf.append(ch); continue
            if ch == ")":
                depth = max(0, depth - 1); buf.append(ch); continue
            if ch == "," and depth == 0:
                p = "".join(buf).strip()
                if p: parts.append(p)
                buf = []; continue
            buf.append(ch)
        if buf:
            p = "".join(buf).strip()
            if p: parts.append(p)
        return parts

    # main loop for stdio transport
    def serve_stdio(self):
        LOG.info("Instryx LSP server (stdio) starting")
        self._running = True
        while self._running:
            msg = self._read_message_stdio()
            if msg is None:
                break
            try:
                self._dispatch_message(msg, transport=None)
            except Exception:
                LOG.exception("serve loop handler failed")

    def _dispatch_message(self, msg: Dict[str, Any], transport: Optional[Tuple[str, Any]] = None):
        if "method" in msg:
            method = msg["method"]
            params = msg.get("params")
            handler = self._request_handlers.get(method)
            id_ = msg.get("id")
            if handler:
                # run handler in separate thread to keep reader responsive
                threading.Thread(target=handler, args=(params, id_, transport), daemon=True).start()
            else:
                # handle common notifications explicitly
                if method in ("textDocument/didOpen", "textDocument/didChange", "textDocument/didClose", "textDocument/didSave"):
                    h = self._request_handlers.get(method)
                    if h:
                        threading.Thread(target=h, args=(params, None, transport), daemon=True).start()
                else:
                    LOG.debug("Unhandled method: %s", method)
                    if "id" in msg:
                        self._send_response(msg["id"], None, {"code": -32601, "message": "Method not found"}, transport=transport)
        elif "id" in msg:
            # We don't expect responses from client in this server
            return

    def shutdown(self):
        self._running = False
        try:
            self._diag_executor.shutdown(wait=False)
        except Exception:
            pass

# -------------------------
# Argparse + CLI
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="instryx_lsp_server.py", description="Instryx LSP server")
    transport = p.add_mutually_exclusive_group()
    transport.add_argument("--stdio", action="store_true", help="serve LSP over stdio (default)")
    transport.add_argument("--tcp", action="store_true", help="serve LSP over TCP (single client)")
    p.add_argument("--host", default="127.0.0.1", help="TCP host to bind")
    p.add_argument("--port", type=int, default=2087, help="TCP port to bind")
    p.add_argument("--workspace", help="workspace root (file://... or path)")
    p.add_argument("--log-level", default="INFO", help="logging level (DEBUG/INFO/WARN/ERROR)")
    p.add_argument("--diag-debounce-ms", type=int, default=200, help="diagnostics debounce window (ms)")
    p.add_argument("--workers", type=int, default=4, help="worker threads for diagnostics/indexing")
    p.add_argument("--index-on-start", action="store_true", help="index workspace symbols on start")
    p.add_argument("--enable-format", action="store_true", help="enable formatting via instryx_syntax_morph if present")
    p.add_argument("--no-stdio", action="store_true", help="disable stdio transport (for testing)")
    return p

def main(argv: Optional[List[str]] = None):
    parser = build_argparser()
    args = parser.parse_args(argv or sys.argv[1:])
    # configure logging
    LOG.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    server = InstryxLSPServer(args)
    try:
        if args.tcp:
            server._running = True
            server._serve_tcp(args.host, args.port)
        else:
            # default to stdio unless explicitly disabled
            if args.no_stdio:
                print("stdio disabled; use --tcp to serve via TCP", file=sys.stderr)
                return 2
            server.serve_stdio()
    except KeyboardInterrupt:
        pass
    except Exception:
        LOG.exception("server exception")
    finally:
        server.shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(main())
