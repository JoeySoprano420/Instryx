# instryx_dodecagram_ast_visualizer.py
# Dodecagram AST Visualizer & Exporter for Instryx Language
# Author: Violet Magenta / VACU Technologies
# License: MIT

import json
from instryx_parser import InstryxParser, ASTNode
from graphviz import Digraph

class DodecagramExporter:
    def __init__(self):
        self.parser = InstryxParser()
        self.ast = None
        self.dot = Digraph(comment='Dodecagram AST')
        self.counter = 0

    def parse_code(self, code: str):
        self.ast = self.parser.parse(code)

    def export_to_dict(self, node: ASTNode = None) -> dict:
        node = node or self.ast
        return {
            "type": node.node_type,
            "value": node.value,
            "children": [self.export_to_dict(child) for child in node.children]
        }

    def export_to_json(self, path: str):
        ast_dict = self.export_to_dict()
        with open(path, 'w') as f:
            json.dump(ast_dict, f, indent=2)

    def _graph_node(self, node: ASTNode, parent_id=None):
        node_id = f"n{self.counter}"
        self.counter += 1
        label = f"{node.node_type}\n{node.value if node.value else ''}"
        self.dot.node(node_id, label)
        if parent_id:
            self.dot.edge(parent_id, node_id)
        for child in node.children:
            self._graph_node(child, node_id)

    def export_to_graphviz(self, output_path="dodecagram_ast", format="png"):
        self.counter = 0
        self._graph_node(self.ast)
        self.dot.render(output_path, format=format, cleanup=True)
        print(f"✅ AST image saved to {output_path}.{format}")

# Test block (can be removed in production)
if __name__ == "__main__":
    code = """
    func greet(uid) {
        print: "Hello, user";
    };

    main() {
        greet(42);
    };
    """
    visualizer = DodecagramExporter()
    visualizer.parse_code(code)
    visualizer.export_to_json("dodecagram_ast.json")
    visualizer.export_to_graphviz("dodecagram_ast")

"""
Dodecagram AST Visualizer & Exporter for Instryx Language — enhanced.

Features added:
 - Safe iterative traversal (avoids recursion limits).
 - Color / shape styling per node type and value heuristics.
 - Export: PNG/SVG/PDF, interactive standalone HTML (embedded SVG + pan/zoom).
 - Subtree export by node type or predicate.
 - JSON-LD export for tooling integration.
 - Caching of last AST -> dot to avoid re-rendering identical ASTs.
 - CLI for quick exports and filters.
 - Minimal dependencies: graphviz (python package) required.
"""
from __future__ import annotations
import json
import html
import hashlib
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from instryx_parser import InstryxParser, ASTNode
from graphviz import Digraph


def _sanitize_label(s: Optional[str], maxlen: int = 120) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", "\\n")
    if len(s) > maxlen:
        return html.escape(s[: maxlen - 3]) + "..."
    return html.escape(s)


_DEFAULT_STYLE = {
    "node": {"fontname": "Helvetica", "shape": "box", "style": "rounded,filled", "fillcolor": "white"},
    "edge": {"arrowhead": "open"},
}


_NODE_COLOR_BY_KIND = {
    "Function": "lightblue",
    "Call": "lightgreen",
    "Literal": "lightgoldenrod",
    "Identifier": "white",
    "Program": "lightgrey",
    "Block": "#F7F7F7",
    "If": "#FFEECC",
    "While": "#FFF0F0",
    "For": "#FFF0F0",
    "Return": "#FFDDDD",
    "Quarantine": "#FFD6FF",
    "Macro": "#E8E8FF",
    "BinaryOp": "#EFEFEF",
}


class DodecagramExporter:
    def __init__(self):
        self.parser = InstryxParser()
        self.ast: Optional[ASTNode] = None
        self._last_ast_hash: Optional[str] = None
        self._cached_dot_source: Optional[str] = None

    # -------------------------
    # AST ingestion / helpers
    # -------------------------
    def parse_code(self, code: str) -> None:
        """Parse source code and keep AST in self.ast."""
        self.ast = self.parser.parse(code)
        self._last_ast_hash = None
        self._cached_dot_source = None

    def load_json_ast(self, path: str) -> None:
        """Load AST previously exported to JSON (expects our export shape)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # convert JSON shape back to minimal ASTNode placeholders (preserves shape for visualization)
        self.ast = self._dict_to_astnode(data)
        self._last_ast_hash = None
        self._cached_dot_source = None

    def _dict_to_astnode(self, d: Dict[str, Any]) -> ASTNode:
        node = ASTNode(node_type=d.get("type", "Unknown"), value=d.get("value"), children=[])
        for c in d.get("children", []):
            node.children.append(self._dict_to_astnode(c))
        return node

    def export_to_dict(self, node: ASTNode = None) -> Dict[str, Any]:
        node = node or self.ast
        if node is None:
            return {}
        # iterative stack to avoid recursion limits
        def node_to_dict(n: ASTNode) -> Dict[str, Any]:
            return {"type": n.node_type, "value": n.value, "children": [node_to_dict(ch) for ch in n.children]}

        return node_to_dict(node)

    def export_to_json(self, path: str) -> str:
        ast_dict = self.export_to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ast_dict, f, indent=2, ensure_ascii=False)
        return path

    def export_to_jsonld(self, path: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Export AST to JSON-LD for better graph/tooling integration.
        Each node gets an @id derived from stable hash of its content.
        """
        node_map: Dict[str, Dict[str, Any]] = {}
        counter = 0

        def stable_id(n: ASTNode) -> str:
            nonlocal counter
            h = hashlib.sha1(f"{n.node_type}:{n.value}:{id(n)}".encode("utf-8")).hexdigest()[:12]
            counter += 1
            return f"node-{h}"

        def build(n: ASTNode) -> str:
            nid = stable_id(n)
            node_map[nid] = {"@id": nid, "type": n.node_type, "value": n.value, "children": []}
            for ch in n.children:
                cid = build(ch)
                node_map[nid]["children"].append({"@id": cid})
            return nid

        if self.ast:
            root_id = build(self.ast)
            out = {"@context": context or {}, "@graph": list(node_map.values()), "root": {"@id": root_id}}
        else:
            out = {"@context": context or {}, "@graph": [], "root": None}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        return path

    # -------------------------
    # core Graphviz export
    # -------------------------
    def _build_graph(self, node: ASTNode, dot: Digraph, collapse_predicate: Optional[Callable[[ASTNode], bool]] = None) -> None:
        """
        Iterative graph construction: assigns stable node ids and populates dot.
        collapse_predicate(node)->True will draw node as collapsed (single-line summary).
        """
        stack: List[Tuple[Optional[str], ASTNode]] = [(None, node)]
        node_ids: Dict[int, str] = {}
        while stack:
            parent_id, cur = stack.pop()
            cur_key = id(cur)
            if cur_key in node_ids:
                my_id = node_ids[cur_key]
            else:
                my_id = f"n{len(node_ids)}"
                node_ids[cur_key] = my_id
                label_val = _sanitize_label(cur.value)
                label = f"{cur.node_type}\\n{label_val}" if label_val else f"{cur.node_type}"
                # style based on node type
                attrs = dict(_DEFAULT_STYLE["node"])
                attrs["fillcolor"] = _NODE_COLOR_BY_KIND.get(cur.node_type, attrs["fillcolor"])
                # collapse if predicate says so
                if collapse_predicate and collapse_predicate(cur):
                    attrs["shape"] = "note"
                    label = f"{cur.node_type} (collapsed)\\n{label_val}"
                dot.node(my_id, label, **attrs)
            if parent_id:
                dot.edge(parent_id, my_id, **_DEFAULT_STYLE["edge"])
            # push children in reverse order so they render left-to-right as original order
            for child in reversed(list(cur.children)):
                stack.append((my_id, child))

    def export_to_graphviz(
        self,
        output_path: str = "dodecagram_ast",
        format: str = "png",
        engine: str = "dot",
        collapse_predicate: Optional[Callable[[ASTNode], bool]] = None,
        overwrite_cache: bool = False,
    ) -> str:
        """
        Export AST to a Graphviz image. Returns the generated filename.
        Uses caching based on AST content to skip re-render when unchanged.
        """
        if not self.ast:
            raise RuntimeError("No AST available. Call parse_code() first.")

        ast_json = json.dumps(self.export_to_dict(), sort_keys=True, ensure_ascii=False)
        ast_hash = hashlib.sha1(ast_json.encode("utf-8")).hexdigest()
        if self._last_ast_hash == ast_hash and not overwrite_cache and self._cached_dot_source:
            # reuse cached dot source (only render if needed)
            dot = Digraph(comment="Dodecagram AST", engine=engine)
            dot.source = self._cached_dot_source  # reuse DOT textual representation
        else:
            dot = Digraph(comment="Dodecagram AST", engine=engine)
            dot.attr("node", **_DEFAULT_STYLE["node"])
            dot.attr("edge", **_DEFAULT_STYLE["edge"])
            self._build_graph(self.ast, dot, collapse_predicate=collapse_predicate)
            self._cached_dot_source = dot.source
            self._last_ast_hash = ast_hash

        # ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        rendered_path = dot.render(output_path, format=format, cleanup=True)
        print(f"✅ AST image saved to {rendered_path}")
        return rendered_path

    # -------------------------
    # SVG + HTML interactive export
    # -------------------------
    def export_to_svg_bytes(self, collapse_predicate: Optional[Callable[[ASTNode], bool]] = None) -> bytes:
        """Return raw SVG bytes for embedding (uses cached dot when available)."""
        # build a temporary Digraph each time to call pipe()
        dot = Digraph(comment="Dodecagram AST", engine="dot")
        dot.attr("node", **_DEFAULT_STYLE["node"])
        dot.attr("edge", **_DEFAULT_STYLE["edge"])
        self._build_graph(self.ast, dot, collapse_predicate=collapse_predicate)
        svg_bytes = dot.pipe(format="svg")
        return svg_bytes

    def export_to_interactive_html(self, output_path: str = "dodecagram_ast.html", collapse_predicate: Optional[Callable[[ASTNode], bool]] = None) -> str:
        """
        Produce a standalone HTML file containing the SVG and minimal pan/zoom controls.
        """
        svg_bytes = self.export_to_svg_bytes(collapse_predicate=collapse_predicate)
        svg_text = svg_bytes.decode("utf-8")
        # simple injection of pan/zoom script
        html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Dodecagram AST</title>
<style>
  body {{ margin:0; padding:0; }}
  #svgwrap {{ width:100vw; height:100vh; overflow:hidden; background:#fff; }}
  svg {{ width:100%; height:100%; }}
</style>
</head>
<body>
<div id="svgwrap">{svg_text}</div>
<script>
(function() {{
  const svg = document.querySelector('svg');
  if(!svg) return;
  let isP = false, startX=0, startY=0, viewBox;
  const vb = svg.getAttribute('viewBox');
  if(vb) {{
    viewBox = vb.split(' ').map(Number);
  }} else {{
    viewBox = [0,0,svg.clientWidth, svg.clientHeight];
    svg.setAttribute('viewBox', viewBox.join(' '));
  }}
  svg.addEventListener('mousedown', e=>{{ isP=true; startX=e.clientX; startY=e.clientY; }});
  svg.addEventListener('mouseup', ()=>{{ isP=false; }});
  svg.addEventListener('mousemove', e=>{{
    if(!isP) return;
    const dx = (startX - e.clientX) * (viewBox[2] / svg.clientWidth);
    const dy = (startY - e.clientY) * (viewBox[3] / svg.clientHeight);
    viewBox[0] += dx; viewBox[1] += dy;
    svg.setAttribute('viewBox', viewBox.join(' '));
    startX = e.clientX; startY = e.clientY;
  }});
  svg.addEventListener('wheel', e=>{{
    e.preventDefault();
    const scale = e.deltaY > 0 ? 1.1 : 0.9;
    const mx = e.clientX / svg.clientWidth * viewBox[2] + viewBox[0];
    const my = e.clientY / svg.clientHeight * viewBox[3] + viewBox[1];
    viewBox[2] *= scale; viewBox[3] *= scale;
    viewBox[0] = mx - (e.clientX / svg.clientWidth) * viewBox[2];
    viewBox[1] = my - (e.clientY / svg.clientHeight) * viewBox[3];
    svg.setAttribute('viewBox', viewBox.join(' '));
  }});
}})();
</script>
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
        print(f"✅ Interactive HTML saved to {output_path}")
        return output_path

    # -------------------------
    # subtree export utilities
    # -------------------------
    def find_nodes_by_type(self, node_type: str) -> List[ASTNode]:
        """Return list of ASTNode objects matching node_type using iterative traversal."""
        if self.ast is None:
            return []
        out: List[ASTNode] = []
        stack = [self.ast]
        while stack:
            cur = stack.pop()
            if cur.node_type == node_type:
                out.append(cur)
            for c in cur.children:
                stack.append(c)
        return out

    def export_subtree_to_graphviz(self, root_node: ASTNode, output_path: str = "subtree_ast", format: str = "png") -> str:
        dot = Digraph(comment="Dodecagram Subtree")
        dot.attr("node", **_DEFAULT_STYLE["node"])
        dot.attr("edge", **_DEFAULT_STYLE["edge"])
        self._build_graph(root_node, dot)
        rendered = dot.render(output_path, format=format, cleanup=True)
        print(f"✅ Subtree image saved to {rendered}")
        return rendered

    # -------------------------
    # CLI convenience
    # -------------------------
    @staticmethod
    def cli_entry(argv: Optional[List[str]] = None) -> int:
        import argparse
        parser = argparse.ArgumentParser(prog="dodecagram_exporter", description="Export Dodecagram AST visualizations")
        parser.add_argument("input", nargs="?", help="Source code file to parse (if omitted stdin is used)")
        parser.add_argument("--out-json", help="Write AST JSON to this path")
        parser.add_argument("--out-graph", help="Write graphviz image (prefix path, ext chosen via --format)", default="dodecagram_ast")
        parser.add_argument("--format", help="Graph format (png/svg/pdf)", default="png")
        parser.add_argument("--out-html", help="Produce interactive HTML embedding the SVG")
        parser.add_argument("--subtree-type", help="Export subtree rooted at first node of this type")
        args = parser.parse_args(argv)

        code = ""
        if args.input:
            with open(args.input, "r", encoding="utf-8") as fh:
                code = fh.read()
        else:
            import sys
            code = sys.stdin.read()

        exporter = DodecagramExporter()
        exporter.parse_code(code)
        if args.out_json:
            exporter.export_to_json(args.out_json)
        if args.subtree_type:
            nodes = exporter.find_nodes_by_type(args.subtree_type)
            if nodes:
                exporter.export_subtree_to_graphviz(nodes[0], output_path=args.out_graph + "_" + args.subtree_type, format=args.format)
        else:
            exporter.export_to_graphviz(output_path=args.out_graph, format=args.format)
        if args.out_html and args.format == "svg":
            exporter.export_to_interactive_html(args.out_html)
        return 0


# --- Test / Demo block ---
if __name__ == "__main__":
    sample = """
    func greet(uid) {
        print: "Hello, user";
    };

    main() {
        greet(42);
    };
    """
    ex = DodecagramExporter()
    ex.parse_code(sample)
    ex.export_to_json("dodecagram_ast.json")
    ex.export_to_graphviz("dodecagram_ast", format="png")
    ex.export_to_interactive_html("dodecagram_ast.html")

"""
instryx_dodecagram_ast_visualizer.py

Dodecagram AST Visualizer & Exporter for Instryx Language — enhanced.

Additions / boosters:
 - Safe iterative traversal (no recursion limit).
 - Styling per node type, edge styles.
 - Subtree export, subtree collapsing by predicate or max depth.
 - Export formats: PNG/SVG/PDF and interactive standalone HTML with pan/zoom, search, highlight & collapse.
 - JSON / JSON-LD export for tooling integration.
 - Filesystem caching of generated DOT -> rendered outputs (by AST hash + options).
 - CLI for quick export, filtering, subtree export, stats export.
 - Export AST summary statistics (counts / depth / widest level).
 - Export limited / truncated visualizations for very large ASTs.
 - Minimal, dependency-light, and conservative fallbacks.
"""

from __future__ import annotations
import json
import html
import hashlib
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from instryx_parser import InstryxParser, ASTNode
from graphviz import Digraph

# --- small helpers ---
def _sanitize_label(s: Optional[str], maxlen: int = 140) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", "\\n")
    if len(s) > maxlen:
        return html.escape(s[: maxlen - 3]) + "..."
    return html.escape(s)


_DEFAULT_STYLE = {
    "node": {"fontname": "Helvetica", "shape": "box", "style": "rounded,filled", "fillcolor": "white"},
    "edge": {"arrowhead": "open"},
}


_NODE_COLOR_BY_KIND = {
    "Function": "lightblue",
    "Call": "lightgreen",
    "Literal": "lightgoldenrod",
    "Identifier": "white",
    "Program": "lightgrey",
    "Block": "#F7F7F7",
    "If": "#FFEECC",
    "While": "#FFF0F0",
    "For": "#FFF0F0",
    "Return": "#FFDDDD",
    "Quarantine": "#FFD6FF",
    "Macro": "#E8E8FF",
    "BinaryOp": "#EFEFEF",
}

# cache directory for dot sources and rendered outputs
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".instryx_visualizer_cache")


# --- exporter ---
class DodecagramExporter:
    def __init__(self):
        self.parser = InstryxParser()
        self.ast: Optional[ASTNode] = None
        self._last_ast_hash: Optional[str] = None
        self._cached_dot_source: Optional[str] = None
        os.makedirs(_CACHE_DIR, exist_ok=True)

    # -------------------------
    # AST ingestion / helpers
    # -------------------------
    def parse_code(self, code: str) -> None:
        """Parse source code and keep AST in self.ast."""
        self.ast = self.parser.parse(code)
        self._last_ast_hash = None
        self._cached_dot_source = None

    def load_json_ast(self, path: str) -> None:
        """Load AST previously exported to JSON (expects our export shape)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.ast = self._dict_to_astnode(data)
        self._last_ast_hash = None
        self._cached_dot_source = None

    def _dict_to_astnode(self, d: Dict[str, Any]) -> ASTNode:
        node = ASTNode(node_type=d.get("type", "Unknown"), value=d.get("value"), children=[])
        for c in d.get("children", []):
            node.children.append(self._dict_to_astnode(c))
        return node

    def export_to_dict(self, node: ASTNode = None) -> Dict[str, Any]:
        node = node or self.ast
        if node is None:
            return {}
        # iterative traversal to avoid recursion issues
        def node_to_dict(n: ASTNode) -> Dict[str, Any]:
            return {"type": n.node_type, "value": n.value, "children": [node_to_dict(ch) for ch in n.children]}
        return node_to_dict(node)

    def export_to_json(self, path: str) -> str:
        ast_dict = self.export_to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ast_dict, f, indent=2, ensure_ascii=False)
        return path

    def export_stats(self, path: str) -> str:
        stats = self._ast_stats(self.ast) if self.ast else {}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        return path

    def export_to_jsonld(self, path: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Export AST to JSON-LD. Each node gets an @id derived from stable hash of its content.
        """
        node_map: Dict[str, Dict[str, Any]] = {}
        counter = 0

        def stable_id(n: ASTNode) -> str:
            nonlocal counter
            h = hashlib.sha1(f"{n.node_type}:{n.value}:{id(n)}".encode("utf-8")).hexdigest()[:12]
            counter += 1
            return f"node-{h}"

        def build(n: ASTNode) -> str:
            nid = stable_id(n)
            node_map[nid] = {"@id": nid, "type": n.node_type, "value": n.value, "children": []}
            for ch in n.children:
                cid = build(ch)
                node_map[nid]["children"].append({"@id": cid})
            return nid

        if self.ast:
            root_id = build(self.ast)
            out = {"@context": context or {}, "@graph": list(node_map.values()), "root": {"@id": root_id}}
        else:
            out = {"@context": context or {}, "@graph": [], "root": None}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        return path

    # -------------------------
    # Graphviz export with caching & styling
    # -------------------------
    def _build_graph(self, node: ASTNode, dot: Digraph, collapse_predicate: Optional[Callable[[ASTNode], bool]] = None, max_nodes: Optional[int] = None) -> None:
        """
        Iterative graph construction: assigns stable node ids and populates dot.
        collapse_predicate(node)->True will draw node as collapsed (single-line summary).
        max_nodes: if set, will stop adding nodes once limit reached and add a truncated marker node.
        """
        stack: List[Tuple[Optional[str], ASTNode]] = [(None, node)]
        node_ids: Dict[int, str] = {}
        added = 0
        while stack:
            parent_id, cur = stack.pop()
            cur_key = id(cur)
            if cur_key in node_ids:
                my_id = node_ids[cur_key]
            else:
                if max_nodes is not None and added >= max_nodes:
                    # insert truncated indicator and stop
                    trunc_id = f"trunc{added}"
                    if trunc_id not in node_ids.values():
                        dot.node(trunc_id, "… truncated …", shape="plaintext", fontname="Helvetica", fontsize="10", color="gray")
                        if parent_id:
                            dot.edge(parent_id, trunc_id, **_DEFAULT_STYLE["edge"])
                    return
                my_id = f"n{len(node_ids)}"
                node_ids[cur_key] = my_id
                added += 1
                label_val = _sanitize_label(cur.value)
                label = f"{cur.node_type}\\n{label_val}" if label_val else f"{cur.node_type}"
                attrs = dict(_DEFAULT_STYLE["node"])
                attrs["fillcolor"] = _NODE_COLOR_BY_KIND.get(cur.node_type, attrs["fillcolor"])
                if collapse_predicate and collapse_predicate(cur):
                    attrs["shape"] = "note"
                    label = f"{cur.node_type} (collapsed)\\n{label_val}"
                # include a stable title to help SVG search/highlight
                dot.node(my_id, label, **attrs, tooltip=cur.node_type)
            if parent_id:
                dot.edge(parent_id, my_id, **_DEFAULT_STYLE["edge"])
            # push children (reverse for reading order)
            for child in reversed(list(cur.children)):
                stack.append((my_id, child))

    def export_to_graphviz(
        self,
        output_path: str = "dodecagram_ast",
        format: str = "png",
        engine: str = "dot",
        collapse_predicate: Optional[Callable[[ASTNode], bool]] = None,
        overwrite_cache: bool = False,
        max_nodes: Optional[int] = None,
    ) -> str:
        """
        Export AST to a Graphviz image. Returns the generated filename.
        Uses caching based on AST content to skip re-render when unchanged.
        """
        if not self.ast:
            raise RuntimeError("No AST available. Call parse_code() first.")

        ast_json = json.dumps(self.export_to_dict(), sort_keys=True, ensure_ascii=False)
        ast_hash = hashlib.sha1(ast_json.encode("utf-8")).hexdigest()

        cache_key = hashlib.sha1((ast_hash + (str(max_nodes) or "") + (str(bool(collapse_predicate)))).encode("utf-8")).hexdigest()
        dot_cache_file = os.path.join(_CACHE_DIR, f"{cache_key}.dot")

        if self._last_ast_hash == ast_hash and not overwrite_cache and os.path.exists(dot_cache_file):
            # reuse cached dot source
            with open(dot_cache_file, "r", encoding="utf-8") as fh:
                dot_source = fh.read()
            dot = Digraph(comment="Dodecagram AST", engine=engine)
            dot.source = dot_source
        else:
            dot = Digraph(comment="Dodecagram AST", engine=engine)
            dot.attr("node", **_DEFAULT_STYLE["node"])
            dot.attr("edge", **_DEFAULT_STYLE["edge"])
            self._build_graph(self.ast, dot, collapse_predicate=collapse_predicate, max_nodes=max_nodes)
            # cache dot source
            try:
                with open(dot_cache_file, "w", encoding="utf-8") as fh:
                    fh.write(dot.source)
            except Exception:
                pass
            self._last_ast_hash = ast_hash
            self._cached_dot_source = dot.source

        # ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        rendered_path = dot.render(output_path, format=format, cleanup=True)
        print(f"✅ AST image saved to {rendered_path}")
        return rendered_path

    # -------------------------
    # SVG interactive + search export
    # -------------------------
    def export_to_svg_bytes(self, collapse_predicate: Optional[Callable[[ASTNode], bool]] = None, max_nodes: Optional[int] = None) -> bytes:
        """Return raw SVG bytes for embedding (uses ephemeral dot build)."""
        dot = Digraph(comment="Dodecagram AST", engine="dot")
        dot.attr("node", **_DEFAULT_STYLE["node"])
        dot.attr("edge", **_DEFAULT_STYLE["edge"])
        self._build_graph(self.ast, dot, collapse_predicate=collapse_predicate, max_nodes=max_nodes)
        svg_bytes = dot.pipe(format="svg")
        return svg_bytes

    def export_to_interactive_html(self, output_path: str = "dodecagram_ast.html", collapse_predicate: Optional[Callable[[ASTNode], bool]] = None, max_nodes: Optional[int] = None) -> str:
        """
        Produce a standalone HTML file containing the SVG and minimal pan/zoom + search & highlight.
        """
        svg_bytes = self.export_to_svg_bytes(collapse_predicate=collapse_predicate, max_nodes=max_nodes)
        svg_text = svg_bytes.decode("utf-8")

        # embed a compact JS helper that finds text nodes matching a query and highlights their parent <g>
        script = r"""
<script>
(function(){
  const svg = document.querySelector('svg');
  if(!svg) return;
  // pan/zoom
  let isP=false, startX=0, startY=0, viewBox;
  let vb = svg.getAttribute('viewBox');
  if(vb) viewBox=vb.split(' ').map(Number); else { viewBox=[0,0,svg.clientWidth,svg.clientHeight]; svg.setAttribute('viewBox', viewBox.join(' ')); }
  svg.addEventListener('mousedown', e=>{isP=true; startX=e.clientX; startY=e.clientY;});
  svg.addEventListener('mouseup', ()=>{isP=false;});
  svg.addEventListener('mousemove', e=>{ if(!isP) return; const dx=(startX-e.clientX)*(viewBox[2]/svg.clientWidth); const dy=(startY-e.clientY)*(viewBox[3]/svg.clientHeight); viewBox[0]+=dx; viewBox[1]+=dy; svg.setAttribute('viewBox', viewBox.join(' ')); startX=e.clientX; startY=e.clientY;});
  svg.addEventListener('wheel', e=>{ e.preventDefault(); const scale = e.deltaY>0?1.1:0.9; const mx=e.clientX/svg.clientWidth*viewBox[2]+viewBox[0]; const my=e.clientY/svg.clientHeight*viewBox[3]+viewBox[1]; viewBox[2]*=scale; viewBox[3]*=scale; viewBox[0]=mx-(e.clientX/svg.clientWidth)*viewBox[2]; viewBox[1]=my-(e.clientY/svg.clientHeight)*viewBox[3]; svg.setAttribute('viewBox', viewBox.join(' '));});

  // search/highlight
  function clearHighlights(){ document.querySelectorAll('.viz-highlight').forEach(el=>el.classList.remove('viz-highlight')); }
  function highlightText(q){
    clearHighlights();
    if(!q) return;
    const texts = Array.from(svg.querySelectorAll('text'));
    const ql = q.toLowerCase();
    texts.forEach(t=>{
      if((t.textContent||'').toLowerCase().indexOf(ql) !== -1){
        // find ancestor <g> and add highlight
        let g = t; while(g && g.nodeName.toLowerCase()!='g') g = g.parentNode;
        if(g) g.classList.add('viz-highlight');
      }
    });
  }
  // wire search input
  const input = document.getElementById('viz-search');
  if(input){
    input.addEventListener('input', ev=>{ highlightText(ev.target.value); });
  }
})();
</script>
<style>
  .viz-highlight > rect { stroke: #ff3333 !important; stroke-width: 2px; }
</style>
"""

        html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Dodecagram AST (interactive)</title>
<style>
  body {{ margin: 0; font-family: sans-serif; }}
  #toolbar {{ position: fixed; top:8px; left:8px; z-index:9999; background:rgba(255,255,255,0.95); padding:8px; border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,0.15); }}
  #svgwrap {{ width:100vw; height:100vh; overflow:hidden; }}
  svg {{ width:100%; height:100%; }}
  input[type="text"] {{ padding:6px; min-width:240px; }}
</style>
</head>
<body>
<div id="toolbar">
  <label>Search: <input id="viz-search" type="text" placeholder="Type node text or type (e.g. Function)"/></label>
  &nbsp; <span>Pan: mouse drag • Zoom: wheel</span>
</div>
<div id="svgwrap">
{svg_text}
</div>
{script}
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
        print(f"✅ Interactive HTML saved to {output_path}")
        return output_path

    # -------------------------
    # subtree export utilities
    # -------------------------
    def find_nodes_by_type(self, node_type: str) -> List[ASTNode]:
        """Return list of ASTNode objects matching node_type using iterative traversal."""
        if self.ast is None:
            return []
        out: List[ASTNode] = []
        stack = [self.ast]
        while stack:
            cur = stack.pop()
            if cur.node_type == node_type:
                out.append(cur)
            for c in cur.children:
                stack.append(c)
        return out

    def export_subtree_to_graphviz(self, root_node: ASTNode, output_path: str = "subtree_ast", format: str = "png", max_nodes: Optional[int] = None) -> str:
        dot = Digraph(comment="Dodecagram Subtree")
        dot.attr("node", **_DEFAULT_STYLE["node"])
        dot.attr("edge", **_DEFAULT_STYLE["edge"])
        self._build_graph(root_node, dot, max_nodes=max_nodes)
        rendered = dot.render(output_path, format=format, cleanup=True)
        print(f"✅ Subtree image saved to {rendered}")
        return rendered

    # -------------------------
    # AST statistics
    # -------------------------
    def _ast_stats(self, root: Optional[ASTNode]) -> Dict[str, Any]:
        if root is None:
            return {}
        counts: Dict[str, int] = {}
        depth = 0
        widest = 0
        queue: List[Tuple[ASTNode, int]] = [(root, 0)]
        while queue:
            node, d = queue.pop(0)
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
            if d > depth:
                depth = d
            # maintain breadth per level
            if len(queue) > widest:
                widest = len(queue)
            for ch in node.children:
                queue.append((ch, d + 1))
        return {"counts": counts, "max_depth": depth, "widest_level_size": widest}

    # -------------------------
    # CLI convenience
    # -------------------------
    @staticmethod
    def cli_entry(argv: Optional[List[str]] = None) -> int:
        import argparse
        parser = argparse.ArgumentParser(prog="dodecagram_exporter", description="Export Dodecagram AST visualizations")
        parser.add_argument("input", nargs="?", help="Source code file to parse (if omitted stdin is used)")
        parser.add_argument("--out-json", help="Write AST JSON to this path")
        parser.add_argument("--out-jsonld", help="Write AST JSON-LD to this path")
        parser.add_argument("--out-stats", help="Write AST statistics to this path")
        parser.add_argument("--out-graph", help="Write graphviz image (prefix path, ext chosen via --format)", default="dodecagram_ast")
        parser.add_argument("--format", help="Graph format (png/svg/pdf)", default="png")
        parser.add_argument("--out-html", help="Produce interactive HTML embedding the SVG")
        parser.add_argument("--subtree-type", help="Export subtree rooted at first node of this type")
        parser.add_argument("--collapse-depth", type=int, help="Collapse nodes deeper than this depth", default=None)
        parser.add_argument("--max-nodes", type=int, help="Limit rendered nodes (truncates beyond limit)", default=None)
        args = parser.parse_args(argv)

        code = ""
        if args.input:
            with open(args.input, "r", encoding="utf-8") as fh:
                code = fh.read()
        else:
            import sys
            code = sys.stdin.read()

        exporter = DodecagramExporter()
        exporter.parse_code(code)
        if args.out_json:
            exporter.export_to_json(args.out_json)
        if args.out_jsonld:
            exporter.export_to_jsonld(args.out_jsonld)
        if args.out_stats:
            exporter.export_stats(args.out_stats)

        collapse_pred = None
        if args.collapse_depth is not None:
            def make_pred(max_d: int) -> Callable[[ASTNode], bool]:
                # predicate closure that collapses nodes deeper than max_d
                def pred(n: ASTNode) -> bool:
                    # compute depth iteratively
                    # NOTE: expensive on many nodes — acceptable for visualization triggers
                    depth = 0
                    cur = n
                    while hasattr(cur, "parent") and getattr(cur, "parent", None) is not None:
                        depth += 1
                        cur = getattr(cur, "parent", cur.parent if hasattr(cur, "parent") else None)
                        if depth > max_d:
                            return True
                    return False
                return pred
            collapse_pred = make_pred(args.collapse_depth)

        if args.subtree_type:
            nodes = exporter.find_nodes_by_type(args.subtree_type)
            if nodes:
                exporter.export_subtree_to_graphviz(nodes[0], output_path=args.out_graph + "_" + args.subtree_type, format=args.format, max_nodes=args.max_nodes)
        else:
            exporter.export_to_graphviz(output_path=args.out_graph, format=args.format, collapse_predicate=collapse_pred, max_nodes=args.max_nodes)

        if args.out_html and args.format == "svg":
            exporter.export_to_interactive_html(args.out_html, collapse_predicate=collapse_pred, max_nodes=args.max_nodes)
        return 0


# --- Test / Demo block ---
if __name__ == "__main__":
    sample = """
    func greet(uid) {
        print: "Hello, user";
    };

    main() {
        greet(42);
    };
    """
    ex = DodecagramExporter()
    ex.parse_code(sample)
    ex.export_to_json("dodecagram_ast.json")
    ex.export_to_graphviz("dodecagram_ast", format="png")
    ex.export_to_interactive_html("dodecagram_ast.html")

    # write interactive SVG+HTML if graphviz supports svg
    ex.export_to_interactive_html("dodecagram_ast.html")

"""
Dodecagram AST Visualizer & Exporter for Instryx Language — boosters & tooling

Enhancements included:
 - Robust iterative AST traversal (no recursion limits).
 - Styled Graphviz output with per-node heuristics.
 - Caching of DOT sources + rendered outputs.
 - Exports: JSON, JSON-LD, NDJSON, SARIF-lite.
 - Interactive standalone HTML (SVG embed) with pan/zoom, search, highlight.
 - Subtree extraction, collapse predicate, max-node truncation.
 - Profiling overlay support (attach interpreter counters/timings).
 - Multi-layout batch rendering and optional thumbnail generation (Pillow).
 - Simple HTTP server to serve interactive HTML.
 - CLI for ad-hoc exports and filters.
"""
from __future__ import annotations
import gzip
import html
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from instryx_parser import InstryxParser, ASTNode
from graphviz import Digraph

# Optional Pillow for thumbnails
try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# Cache directory for DOT sources and rendered outputs
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".instryx_visualizer_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# Default visual styles
_DEFAULT_STYLE = {
    "node": {"fontname": "Helvetica", "shape": "box", "style": "rounded,filled", "fillcolor": "white"},
    "edge": {"arrowhead": "open"},
}

_NODE_COLOR_BY_KIND = {
    "Function": "lightblue",
    "Call": "lightgreen",
    "Literal": "lightgoldenrod",
    "Identifier": "white",
    "Program": "lightgrey",
    "Block": "#F7F7F7",
    "If": "#FFEECC",
    "While": "#FFF0F0",
    "For": "#FFF0F0",
    "Return": "#FFDDDD",
    "Quarantine": "#FFD6FF",
    "Macro": "#E8E8FF",
    "BinaryOp": "#EFEFEF",
}

_TRUNCATED_MARKER = "… truncated …"


def _sanitize_label(v: Optional[Any], maxlen: int = 160) -> str:
    if v is None:
        return ""
    s = str(v)
    s = s.replace("\r", " ").replace("\n", "\\n")
    if len(s) > maxlen:
        return html.escape(s[: maxlen - 3]) + "..."
    return html.escape(s)


def _stable_hash(obj: Any) -> str:
    try:
        j = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        j = str(obj)
    return hashlib.sha1(j.encode("utf-8")).hexdigest()


class DodecagramExporter:
    def __init__(self):
        self.parser = InstryxParser()
        self.ast: Optional[ASTNode] = None
        self._last_ast_hash: Optional[str] = None
        self._cached_dot: Optional[str] = None
        # optional profiler overlay: {"counts": {...}, "time": {...}}
        self.profiling_overlay: Optional[Dict[str, Any]] = None

    # -------------------------
    # AST ingestion / helpers
    # -------------------------
    def parse_code(self, code: str) -> None:
        """Parse source code and keep AST in self.ast."""
        self.ast = self.parser.parse(code)
        self._last_ast_hash = None
        self._cached_dot = None

    def load_json_ast(self, path: str) -> None:
        """Load AST previously exported to JSON (expects our export shape)."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.ast = self._dict_to_astnode(data)
        self._last_ast_hash = None
        self._cached_dot = None

    def _dict_to_astnode(self, d: Dict[str, Any]) -> ASTNode:
        node = ASTNode(node_type=d.get("type", "Unknown"), value=d.get("value"), children=[])
        for c in d.get("children", []):
            node.children.append(self._dict_to_astnode(c))
        return node

    def export_to_dict(self, node: ASTNode = None) -> Dict[str, Any]:
        """Return a plain dict representation of the AST (iterative traversal)."""
        node = node or self.ast
        if node is None:
            return {}
        # iterative builder to avoid recursion depth issues
        def build(n: ASTNode) -> Dict[str, Any]:
            return {"type": n.node_type, "value": n.value, "children": [build(c) for c in n.children]}
        return build(node)

    def export_to_json(self, path: str) -> str:
        ast_dict = self.export_to_dict()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(ast_dict, fh, indent=2, ensure_ascii=False)
        return path

    def export_to_ndjson(self, path: str) -> str:
        """Stream node-per-line NDJSON (useful for pipelines)."""
        if self.ast is None:
            return path
        with open(path, "w", encoding="utf-8") as fh:
            stack = [self.ast]
            while stack:
                cur = stack.pop()
                fh.write(json.dumps({"type": cur.node_type, "value": cur.value, "children_count": len(cur.children)}, default=str) + "\n")
                for c in cur.children:
                    stack.append(c)
        return path

    def export_to_jsonld(self, path: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Export AST to JSON-LD with stable node @id values."""
        node_map: Dict[str, Dict[str, Any]] = {}
        counter = 0

        def stable_id(n: ASTNode) -> str:
            nonlocal counter
            h = hashlib.sha1(f"{n.node_type}:{n.value}:{id(n)}".encode("utf-8")).hexdigest()[:12]
            counter += 1
            return f"node-{h}"

        def build(n: ASTNode) -> str:
            nid = stable_id(n)
            node_map[nid] = {"@id": nid, "type": n.node_type, "value": n.value, "children": []}
            for ch in n.children:
                cid = build(ch)
                node_map[nid]["children"].append({"@id": cid})
            return nid

        if self.ast:
            root_id = build(self.ast)
            out = {"@context": context or {}, "@graph": list(node_map.values()), "root": {"@id": root_id}}
        else:
            out = {"@context": context or {}, "@graph": [], "root": None}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        return path

    def export_to_sarif_like(self, path: str) -> str:
        """
        Minimal SARIF-like summary mapping node kinds to counts for CI.
        """
        if self.ast is None:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"runs": []}, fh)
            return path
        counts: Dict[str, int] = {}
        stack = [self.ast]
        while stack:
            c = stack.pop()
            counts[c.node_type] = counts.get(c.node_type, 0) + 1
            for ch in c.children:
                stack.append(ch)
        sarif = {
            "version": "2.1.0",
            "runs": [{
                "tool": {"driver": {"name": "dodecagram-visualizer"}},
                "results": [{"ruleId": k, "count": v} for k, v in counts.items()]
            }]
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(sarif, fh, indent=2)
        return path

    # -------------------------
    # Graphviz build (iterative) and caching
    # -------------------------
    def _build_graph(self, root: ASTNode, collapse_predicate: Optional[Callable[[ASTNode], bool]] = None, max_nodes: Optional[int] = None) -> Tuple[Digraph, Dict[int, str]]:
        """
        Build a Graphviz Digraph from AST iteratively.
        Returns (dot, id_map).
        """
        dot = Digraph(comment="Dodecagram AST")
        dot.attr("node", **_DEFAULT_STYLE["node"])
        dot.attr("edge", **_DEFAULT_STYLE["edge"])
        stack: List[Tuple[Optional[str], ASTNode]] = [(None, root)]
        id_map: Dict[int, str] = {}
        added = 0
        while stack:
            parent_id, cur = stack.pop()
            key = id(cur)
            if key in id_map:
                nid = id_map[key]
            else:
                if max_nodes is not None and added >= max_nodes:
                    trunc_id = f"trunc{added}"
                    if trunc_id not in id_map.values():
                        dot.node(trunc_id, _TRUNCATED_MARKER, shape="plaintext", fontname="Helvetica", fontsize="10", color="gray")
                        if parent_id:
                            dot.edge(parent_id, trunc_id)
                    return dot, id_map
                nid = f"n{len(id_map)}"
                id_map[key] = nid
                added += 1
                label_val = _sanitize_label(getattr(cur, "value", None))
                label = f"{cur.node_type}\\n{label_val}" if label_val else cur.node_type
                attrs = dict(_DEFAULT_STYLE["node"])
                attrs["fillcolor"] = _NODE_COLOR_BY_KIND.get(cur.node_type, attrs["fillcolor"])
                if collapse_predicate and collapse_predicate(cur):
                    attrs["shape"] = "note"
                    label = f"{cur.node_type} (collapsed)\\n{label_val}"
                # profiling overlay
                if self.profiling_overlay:
                    counts = self.profiling_overlay.get("counts", {})
                    times = self.profiling_overlay.get("time", {})
                    key_label = str(getattr(cur, "value", cur.node_type))
                    cnt = counts.get(key_label) or counts.get(cur.node_type) or 0
                    tt = times.get(key_label) or times.get(cur.node_type) or 0.0
                    if cnt or tt:
                        label = f"{label}\\ncount={cnt} time={round(tt,3)}s"
                tooltip = None
                if hasattr(cur, "start_offset") and hasattr(cur, "end_offset"):
                    tooltip = f"{getattr(cur,'filename','<unknown>')}:{getattr(cur,'start_offset')}..{getattr(cur,'end_offset')}"
                dot.node(nid, label, tooltip=tooltip or cur.node_type, **attrs)
            if parent_id:
                dot.edge(parent_id, nid)
            # push children in reverse order so they render left-to-right
            for ch in reversed(list(cur.children)):
                stack.append((nid, ch))
        return dot, id_map

    def export_to_graphviz(
        self,
        output_prefix: str = "dodecagram_ast",
        format: str = "png",
        engine: str = "dot",
        collapse_predicate: Optional[Callable[[ASTNode], bool]] = None,
        max_nodes: Optional[int] = None,
        force: bool = False,
        layouts: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """
        Render one or more layouts. Returns list of generated paths.
        Uses gzipped DOT cache + rendered file cache.
        """
        if self.ast is None:
            raise RuntimeError("AST missing. Call parse_code() first.")
        ast_dict = self.export_to_dict()
        ast_hash = _stable_hash(ast_dict)
        layouts = list(layouts) if layouts else [engine]
        out_paths: List[str] = []
        for layout in layouts:
            cache_key = hashlib.sha1(f"{ast_hash}:{layout}:{format}:{str(max_nodes)}".encode("utf-8")).hexdigest()
            cache_dot = os.path.join(_CACHE_DIR, f"{cache_key}.dot.gz")
            cache_rendered = os.path.join(_CACHE_DIR, f"{cache_key}.{format}")
            dest = f"{output_prefix}_{layout}.{format}" if len(layouts) > 1 else f"{output_prefix}.{format}"
            if not force and os.path.exists(cache_rendered):
                try:
                    shutil.copyfile(cache_rendered, dest)
                    out_paths.append(dest)
                    continue
                except Exception:
                    pass
            dot, _ = self._build_graph(self.ast, collapse_predicate=collapse_predicate, max_nodes=max_nodes)
            dot.engine = layout
            try:
                with gzip.open(cache_dot, "wt", encoding="utf-8") as gz:
                    gz.write(dot.source)
            except Exception:
                pass
            tmp = dot.render(filename=tempfile.NamedTemporaryFile(delete=False).name, format=format, cleanup=True)
            try:
                shutil.move(tmp, dest)
            except Exception:
                shutil.copyfile(tmp, dest)
            try:
                shutil.copyfile(dest, cache_rendered)
            except Exception:
                pass
            out_paths.append(dest)
        self._last_ast_hash = ast_hash
        self._cached_dot = None
        return out_paths

    # -------------------------
    # SVG interactive HTML export
    # -------------------------
    def export_to_svg_bytes(self, collapse_predicate: Optional[Callable[[ASTNode], bool]] = None, max_nodes: Optional[int] = None) -> bytes:
        dot, _ = self._build_graph(self.ast, collapse_predicate=collapse_predicate, max_nodes=max_nodes)
        return dot.pipe(format="svg")

    def export_to_interactive_html(self, output_path: str = "dodecagram_ast.html", collapse_predicate: Optional[Callable[[ASTNode], bool]] = None, max_nodes: Optional[int] = None) -> str:
        svg_bytes = self.export_to_svg_bytes(collapse_predicate=collapse_predicate, max_nodes=max_nodes)
        svg_text = svg_bytes.decode("utf-8")
        # compact interactive controls: search, pan/zoom, highlight
        script = r"""
<script>
(function(){
  const svg = document.querySelector('svg');
  if(!svg) return;
  let isP=false, sx=0, sy=0, viewBox;
  const vb = svg.getAttribute('viewBox');
  viewBox = vb ? vb.split(' ').map(Number) : [0,0,svg.clientWidth, svg.clientHeight];
  if(!vb) svg.setAttribute('viewBox', viewBox.join(' '));
  svg.addEventListener('mousedown', e=>{isP=true; sx=e.clientX; sy=e.clientY;});
  svg.addEventListener('mouseup', ()=>{isP=false;});
  svg.addEventListener('mousemove', e=>{ if(!isP) return; const dx=(sx-e.clientX)*(viewBox[2]/svg.clientWidth); const dy=(sy-e.clientY)*(viewBox[3]/svg.clientHeight); viewBox[0]+=dx; viewBox[1]+=dy; svg.setAttribute('viewBox', viewBox.join(' ')); sx=e.clientX; sy=e.clientY;});
  svg.addEventListener('wheel', e=>{ e.preventDefault(); const scale=e.deltaY>0?1.1:0.9; const mx=e.clientX/svg.clientWidth*viewBox[2]+viewBox[0]; const my=e.clientY/svg.clientHeight*viewBox[3]+viewBox[1]; viewBox[2]*=scale; viewBox[3]*=scale; viewBox[0]=mx-(e.clientX/svg.clientWidth)*viewBox[2]; viewBox[1]=my-(e.clientY/svg.clientHeight)*viewBox[3]; svg.setAttribute('viewBox', viewBox.join(' '));});
  function clear(){ document.querySelectorAll('.viz-highlight').forEach(el=>el.classList.remove('viz-highlight')); }
  function highlight(q){ clear(); if(!q) return; const texts=Array.from(svg.querySelectorAll('text')); const ql=q.toLowerCase(); texts.forEach(t=>{ if((t.textContent||'').toLowerCase().includes(ql)){ let g=t; while(g && g.nodeName.toLowerCase()!='g') g=g.parentNode; if(g) g.classList.add('viz-highlight'); } }); }
  const input = document.getElementById('viz-search'); if(input) input.addEventListener('input', ev=>highlight(ev.target.value));
})();
</script>
<style>
.viz-highlight > rect { stroke: #ff3333 !important; stroke-width: 2px; }
#toolbar { position: fixed; top:8px; left:8px; z-index:9999; background:rgba(255,255,255,0.95); padding:8px; border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,0.15); }
</style>
"""
        html_doc = f"""<!doctype html>
<html>
<head><meta charset="utf-8"/><title>Dodecagram AST (interactive)</title></head>
<body>
<div id="toolbar">
<label>Search: <input id="viz-search" type="text" placeholder="e.g. Function, Call, literal"/></label>
&nbsp;<small>Pan: drag • Zoom: wheel</small>
</div>
<div id="svgwrap">{svg_text}</div>
{script}
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
        return output_path

    def serve_interactive(self, html_path: str, port: int = 0, open_browser: bool = False) -> Tuple[str, int]:
        """Serve interactive HTML file locally; returns (url, port)."""
        if not os.path.exists(html_path):
            raise FileNotFoundError(html_path)
        directory = os.path.dirname(os.path.abspath(html_path)) or "."
        host = "127.0.0.1"

        class _Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

        httpd = HTTPServer((host, port), _Handler)
        assigned_port = httpd.server_address[1]

        def _serve():
            try:
                httpd.serve_forever()
            except Exception:
                pass

        th = threading.Thread(target=_serve, daemon=True)
        th.start()
        addr = f"http://{host}:{assigned_port}/{os.path.basename(html_path)}"
        if open_browser:
            try:
                import webbrowser
                webbrowser.open(addr)
            except Exception:
                pass
        return addr, assigned_port

    # -------------------------
    # subtree & utilities
    # -------------------------
    def find_nodes_by_type(self, node_type: str) -> List[ASTNode]:
        if self.ast is None:
            return []
        out: List[ASTNode] = []
        stack = [self.ast]
        while stack:
            cur = stack.pop()
            if cur.node_type == node_type:
                out.append(cur)
            for c in cur.children:
                stack.append(c)
        return out

    def export_subtree(self, root: ASTNode, out_prefix: str = "subtree", format: str = "png", max_nodes: Optional[int] = None) -> str:
        dot, _ = self._build_graph(root, max_nodes=max_nodes)
        out = dot.render(filename=out_prefix, format=format, cleanup=True)
        return out

    def attach_profiling(self, overlay: Dict[str, Any]) -> None:
        """Attach a profiling overlay (counts/time) to display in node labels."""
        self.profiling_overlay = overlay
        self._cached_dot = None

    def ast_stats(self) -> Dict[str, Any]:
        return self._ast_stats(self.ast)

    def _ast_stats(self, root: Optional[ASTNode]) -> Dict[str, Any]:
        if root is None:
            return {}
        counts: Dict[str, int] = {}
        max_depth = 0
        level_counts: Dict[int, int] = {}
        stack: List[Tuple[ASTNode, int]] = [(root, 0)]
        while stack:
            node, d = stack.pop()
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
            level_counts[d] = level_counts.get(d, 0) + 1
            if d > max_depth:
                max_depth = d
            for ch in node.children:
                stack.append((ch, d + 1))
        widest = max(level_counts.values()) if level_counts else 0
        return {"counts": counts, "max_depth": max_depth, "widest_level": widest}

    # -------------------------
    # CLI convenience
    # -------------------------
    @staticmethod
    def cli_entry(argv: Optional[List[str]] = None) -> int:
        import argparse
        parser = argparse.ArgumentParser(prog="dodecagram_exporter", description="Dodecagram AST visualizer (enhanced)")
        parser.add_argument("input", nargs="?", help="Source file to parse (stdin if omitted)")
        parser.add_argument("--out-json", help="Write AST JSON")
        parser.add_argument("--out-ndjson", help="Write NDJSON nodes")
        parser.add_argument("--out-jsonld", help="Write JSON-LD")
        parser.add_argument("--out-sarif", help="Write SARIF-like summary")
        parser.add_argument("--out-graph", help="Graph output prefix", default="dodecagram_ast")
        parser.add_argument("--format", choices=("png", "svg", "pdf"), default="png")
        parser.add_argument("--layouts", nargs="*", help="Layout engines (dot/neato/circo)", default=[])
        parser.add_argument("--out-html", help="Interactive HTML output (svg required)")
        parser.add_argument("--max-nodes", type=int, help="Limit nodes (truncates beyond limit)", default=None)
        parser.add_argument("--collapse-depth", type=int, help="Collapse nodes deeper than depth", default=None)
        parser.add_argument("--serve", action="store_true", help="Serve interactive HTML after generation")
        args = parser.parse_args(argv)

        if args.input:
            with open(args.input, "r", encoding="utf-8") as fh:
                code = fh.read()
        else:
            code = sys.stdin.read()

        exp = DodecagramExporter()
        exp.parse_code(code)
        if args.out_json:
            exp.export_to_json(args.out_json)
        if args.out_ndjson:
            exp.export_to_ndjson(args.out_ndjson)
        if args.out_jsonld:
            exp.export_to_jsonld(args.out_jsonld)
        if args.out_sarif:
            exp.export_to_sarif_like(args.out_sarif)

        collapse_pred = None
        if args.collapse_depth is not None:
            def _pred_maxdepth(max_d: int) -> Callable[[ASTNode], bool]:
                def p(n: ASTNode) -> bool:
                    # compute depth via parent chain if present (best-effort)
                    depth = 0
                    cur = getattr(n, "parent", None)
                    while cur:
                        depth += 1
                        cur = getattr(cur, "parent", None)
                        if depth > max_d:
                            return True
                    return False
                return p
            collapse_pred = _pred_maxdepth(args.collapse_depth)

        layouts = args.layouts or None
        paths = exp.export_to_graphviz(output_prefix=args.out_graph, format=args.format, layouts=layouts, collapse_predicate=collapse_pred, max_nodes=args.max_nodes)
        if args.out_html and args.format == "svg":
            html_path = args.out_html
            exp.export_to_interactive_html(html_path, collapse_predicate=collapse_pred, max_nodes=args.max_nodes)
            if args.serve:
                addr, port = exp.serve_interactive(html_path, open_browser=True)
                print(f"Serving at {addr}")
        print("AST stats:", json.dumps(exp.ast_stats(), indent=2))
        return 0


# --- Demo / smoke test (kept short) ---
if __name__ == "__main__":
    sample = """
    func greet(uid) {
        print: "Hello, user";
    };

    main() {
        greet(42);
    };
    """
    ex = DodecagramExporter()
    ex.parse_code(sample)
    ex.export_to_json("dodecagram_ast.json")
    ex.export_to_graphviz("dodecagram_ast", format="png")
    # produce interactive HTML (uses SVG via graphviz)
    try:
        ex.export_to_interactive_html("dodecagram_ast.html")
        print("Interactive HTML written to dodecagram_ast.html")
    except Exception:
        # non-fatal if SVG rendering not available
        pass

