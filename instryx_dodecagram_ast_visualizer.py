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
