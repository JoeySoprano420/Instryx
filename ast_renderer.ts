/**
 * ast_renderer.ts
 *
 * Complete, production-ready AST -> Instryx source renderer.
 *
 * Features:
 * - Type-safe AST node definitions (common Instryx/Dodecagram shapes).
 * - Deterministic pretty-printer with configurable indentation and formatting.
 * - Compact / minified output mode.
 * - Preserves comments when present on nodes.
 * - Optional lightweight source-map-like mapping (node id -> output range).
 * - Safe string escaping and identifier quoting.
 * - Extensible renderer: add handlers for new node kinds via `Renderer.nodeHandlers`.
 *
 * Usage:
 *   import { renderAst, RenderOptions } from "./ast_renderer";
 *   const out = renderAst(ast, { indent: "  ", compact: false });
 *
 * Notes:
 * - The AST shape is intentionally permissive: nodes are discriminated by `type`.
 * - Nodes may include optional `id` and `loc` fields. If `id` missing, an internal id will be assigned.
 * - Comments, diagnostics, or macro metadata may be present on nodes as `leadingComments` / `trailingComments`.
 */

export type RenderOptions = {
  indent?: string;           // e.g. "  " or "\t"
  compact?: boolean;         // produce minimal whitespace
  maxInlineLength?: number;  // prefer single-line rendering below this length
  preserveComments?: boolean;
  emitSourceMap?: boolean;   // return mapping of node ids -> { start,end }
};

export type RenderResult = {
  code: string;
  sourceMap?: Record<string, { start: number; end: number }>;
};

let __internalNodeId = 1;
function genNodeId(): string {
  return `n${__internalNodeId++}`;
}

/* -------------------------
   AST Node Definitions
   (Common Instryx node shapes; renderer is permissive)
   ------------------------- */

export interface BaseNode {
  type: string;
  id?: string;
  loc?: { start: number; end: number } | null;
  leadingComments?: string[];
  trailingComments?: string[];
  [k: string]: any;
}

export interface ProgramNode extends BaseNode {
  type: "Program";
  body: BaseNode[];
}

export interface FuncNode extends BaseNode {
  type: "Func";
  name: string;
  params: string[];          // parameter names
  body: BlockNode | BaseNode;
  exported?: boolean;
  extern?: boolean;
}

export interface BlockNode extends BaseNode {
  type: "Block";
  stmts: BaseNode[];
}

export interface ReturnNode extends BaseNode {
  type: "Return";
  value?: BaseNode;
}

export interface IfNode extends BaseNode {
  type: "If";
  test: BaseNode;
  consequent: BlockNode;
  alternate?: BlockNode | null;
}

export interface WhileNode extends BaseNode {
  type: "While";
  test: BaseNode;
  body: BlockNode;
}

export interface ForNode extends BaseNode {
  type: "For";
  index?: string;
  from?: BaseNode;
  to?: BaseNode;
  body: BlockNode;
}

export interface AssignNode extends BaseNode {
  type: "Assign";
  target: string | BaseNode;
  value: BaseNode;
}

export interface CallNode extends BaseNode {
  type: "Call";
  callee: string | BaseNode;
  args: BaseNode[];
}

export interface VarNode extends BaseNode {
  type: "Var";
  name: string;
}

export interface LiteralNode extends BaseNode {
  type: "Literal";
  value: string | number | boolean | null | any[];
}

export interface BinaryNode extends BaseNode {
  type: "Binary";
  op: string;
  left: BaseNode;
  right: BaseNode;
}

export interface UnaryNode extends BaseNode {
  type: "Unary";
  op: string;
  expr: BaseNode;
}

export interface CommentNode extends BaseNode {
  type: "Comment";
  text: string;
  multiline?: boolean;
}

export interface QuarantineNode extends BaseNode {
  type: "Quarantine";
  tryBlock: BlockNode;
  replaceBlock?: BlockNode | null;
  eraseBlock?: BlockNode | null;
}

export interface ExternNode extends BaseNode {
  type: "Extern";
  name: string;
  params?: string[];
  abi?: string;
}

export type InstryxNode =
  | ProgramNode | FuncNode | BlockNode | ReturnNode | IfNode | WhileNode | ForNode
  | AssignNode | CallNode | VarNode | LiteralNode | BinaryNode | UnaryNode
  | CommentNode | QuarantineNode | ExternNode | BaseNode;

/* -------------------------
   Renderer
   ------------------------- */

type NodeHandler = (node: BaseNode, state: RendererState) => string;

class RendererState {
  opts: Required<RenderOptions>;
  indentLevel: number;
  // output building
  out: string[];
  // position tracking for simple source-map
  pos: number;
  sourceMap: Record<string, { start: number; end: number }>;
  constructor(opts: RenderOptions) {
    this.opts = {
      indent: opts.indent ?? "  ",
      compact: !!opts.compact,
      maxInlineLength: opts.maxInlineLength ?? 80,
      preserveComments: !!opts.preserveComments,
      emitSourceMap: !!opts.emitSourceMap,
    };
    this.indentLevel = 0;
    this.out = [];
    this.pos = 0;
    this.sourceMap = {};
  }

  push(s: string) {
    this.out.push(s);
    this.pos += s.length;
  }
  pushSpace() {
    if (!this.opts.compact) this.push(" ");
  }
  newline() {
    if (this.opts.compact) return;
    this.push("\n");
  }
  indent() {
    if (!this.opts.compact) this.push(this.opts.indent.repeat(this.indentLevel));
  }
  recordNode(node: BaseNode, startPos: number) {
    if (!this.opts.emitSourceMap) return;
    const id = node.id ?? genNodeId();
    node.id = id;
    this.sourceMap[id] = this.sourceMap[id] || { start: startPos, end: startPos };
  }
  finalizeNode(node: BaseNode, endPos: number) {
    if (!this.opts.emitSourceMap) return;
    const id = node.id!;
    const entry = this.sourceMap[id];
    if (entry) entry.end = endPos;
    else this.sourceMap[id] = { start: 0, end: endPos };
  }
  get code() {
    return this.out.join("");
  }
}

/* -------------------------
   Node handlers
   ------------------------- */

const defaultHandlers: Record<string, NodeHandler> = {};

// helper utilities used by handlers
function escString(s: string): string {
  // minimal escaping
  return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
}

function isSimpleLiteral(node: BaseNode): node is LiteralNode {
  return node.type === "Literal" && (typeof (node as LiteralNode).value === "string" || typeof (node as LiteralNode).value === "number" || typeof (node as LiteralNode).value === "boolean" || (node as LiteralNode).value === null);
}

function inlinePossible(str: string, state: RendererState) {
  return str.length <= state.opts.maxInlineLength && state.opts.compact === false;
}

/* Generic comment rendering */
function renderLeadingComments(node: BaseNode, s: RendererState) {
  if (!s.opts.preserveComments) return;
  const arr = (node.leadingComments ?? []) as string[];
  for (const c of arr) {
    s.indent();
    if (c.includes("\n")) {
      s.push("/* " + c.replace(/\*\//g, "*\\/") + " */");
    } else {
      s.push("-- " + c);
    }
    s.newline();
  }
}

function renderTrailingComments(node: BaseNode, s: RendererState) {
  if (!s.opts.preserveComments) return;
  const arr = (node.trailingComments ?? []) as string[];
  for (const c of arr) {
    s.push(" /* " + c.replace(/\*\//g, "*\\/") + " */");
  }
}

/* Handler: Program */
defaultHandlers["Program"] = (node: ProgramNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  node.body = node.body || [];
  for (let i = 0; i < node.body.length; i++) {
    const child = node.body[i];
    s.indent();
    renderNode(child, s);
    // Ensure statements end with semicolon when appropriate
    if (!s.opts.compact) {
      // some node types include their own terminating semicolons (e.g. Func -> uses trailing semicolon optional)
      // For general statements, append semicolon if not already present
      if (!s.code.endsWith(";") && child.type !== "Func" && child.type !== "Block" && child.type !== "Comment") {
        s.push(";");
      }
    } else {
      // compact: always append semicolon except for Func/Block/Comment
      if (child.type !== "Func" && child.type !== "Block" && child.type !== "Comment") s.push(";");
    }
    s.newline();
  }
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: Func */
defaultHandlers["Func"] = (node: FuncNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  // signature
  if (node.exported) s.push("export ");
  if (node.extern) s.push('extern ');
  s.push("func ");
  s.push(node.name);
  s.push("(" + (node.params || []).join(", ") + ")");
  // body may be Block or other (extern)
  if ((node as any).extern || node.extern) {
    s.push(";");
    s.newline();
    s.finalizeNode(node, s.pos);
    renderTrailingComments(node, s);
    return s.code.slice(start, s.pos);
  }
  // body
  const body = node.body as BlockNode;
  s.push(" ");
  // block rendering
  renderNode(body, s);
  s.finalizeNode(node, s.pos);
  renderTrailingComments(node, s);
  return s.code.slice(start, s.pos);
};

/* Handler: Block */
defaultHandlers["Block"] = (node: BlockNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  s.push("{");
  if (!s.opts.compact) s.newline();
  s.indentLevel++;
  for (let i = 0; i < (node.stmts || []).length; i++) {
    const stmt = node.stmts[i];
    s.indent();
    renderNode(stmt, s);
    // semicolon for statements that are not blocks/functions
    if (!s.opts.compact) {
      if (!s.code.endsWith(";") && stmt.type !== "Func" && stmt.type !== "Block" && stmt.type !== "Comment") s.push(";");
    } else {
      if (stmt.type !== "Func" && stmt.type !== "Block" && stmt.type !== "Comment") s.push(";");
    }
    s.newline();
  }
  s.indentLevel--;
  s.indent();
  s.push("}");
  s.finalizeNode(node, s.pos);
  renderTrailingComments(node, s);
  return s.code.slice(start, s.pos);
};

/* Handler: Return */
defaultHandlers["Return"] = (node: ReturnNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  s.push("return");
  if (node.value !== undefined && node.value !== null) {
    s.push(" ");
    renderNode(node.value, s);
  }
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: If */
defaultHandlers["If"] = (node: IfNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  s.push("if ");
  renderNode(node.test, s);
  s.push(" ");
  renderNode(node.consequent, s);
  if (node.alternate) {
    s.push(" else ");
    renderNode(node.alternate, s);
  }
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: While */
defaultHandlers["While"] = (node: WhileNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  s.push("while ");
  renderNode(node.test, s);
  s.push(" ");
  renderNode(node.body, s);
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: For */
defaultHandlers["For"] = (node: ForNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  s.push("for ");
  if (node.index) {
    s.push(node.index + " ");
  }
  if (node.from || node.to) {
    s.push("in ");
    if (node.from) renderNode(node.from, s);
    s.push("..");
    if (node.to) renderNode(node.to, s);
  }
  s.push(" ");
  renderNode(node.body, s);
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: Assign */
defaultHandlers["Assign"] = (node: AssignNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  if (typeof node.target === "string") {
    s.push(node.target + " = ");
  } else {
    renderNode(node.target as BaseNode, s);
    s.push(" = ");
  }
  renderNode(node.value, s);
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: Call */
defaultHandlers["Call"] = (node: CallNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  const calleeStr = (typeof node.callee === "string") ? node.callee : null;
  if (calleeStr) s.push(calleeStr);
  else renderNode(node.callee as BaseNode, s);
  s.push("(");
  for (let i = 0; i < (node.args || []).length; i++) {
    if (i) s.push(",");
    if (!s.opts.compact) s.push(" ");
    renderNode(node.args[i], s);
  }
  if (!s.opts.compact && (node.args?.length ?? 0) > 0) s.push(" ");
  s.push(")");
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: Var */
defaultHandlers["Var"] = (node: VarNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  s.push(node.name);
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: Literal */
defaultHandlers["Literal"] = (node: LiteralNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  const v = node.value;
  if (typeof v === "string") {
    s.push(`"${escString(v)}"`);
  } else if (typeof v === "number") {
    s.push(String(v));
  } else if (typeof v === "boolean") {
    s.push(v ? "true" : "false");
  } else if (v === null) {
    s.push("null");
  } else if (Array.isArray(v)) {
    s.push("[");
    for (let i = 0; i < v.length; i++) {
      if (i) s.push(", ");
      const el = v[i];
      if (typeof el === "string") s.push(`"${escString(el)}"`);
      else s.push(String(el));
    }
    s.push("]");
  } else {
    // fallback to JSON
    s.push(JSON.stringify(v));
  }
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: Binary */
defaultHandlers["Binary"] = (node: BinaryNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  // Attempt to inline if small
  const savePos = s.pos;
  const leftStart = s.pos;
  renderNode(node.left, s);
  const leftStr = s.code.slice(leftStart, s.pos);
  s.push(" ");
  s.push(node.op);
  s.push(" ");
  const rightStart = s.pos;
  renderNode(node.right, s);
  const rightStr = s.code.slice(rightStart, s.pos);
  const whole = s.code.slice(savePos, s.pos);
  // Don't reformat; default representation is fine
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return whole;
};

/* Handler: Unary */
defaultHandlers["Unary"] = (node: UnaryNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  s.push(node.op);
  if (!node.op.match(/^\w+$/)) s.push("");
  renderNode(node.expr, s);
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: Comment */
defaultHandlers["Comment"] = (node: CommentNode, s: RendererState) => {
  const start = s.pos;
  if (node.multiline) {
    s.push("/* " + (node.text ?? "") + " */");
  } else {
    s.push("-- " + (node.text ?? ""));
  }
  s.newline();
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: Quarantine tri-phase */
defaultHandlers["Quarantine"] = (node: QuarantineNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  s.push("quarantine try ");
  renderNode(node.tryBlock, s);
  s.push(" replace ");
  renderNode(node.replaceBlock ?? { type: "Block", stmts: [] } as BlockNode, s);
  s.push(" erase ");
  renderNode(node.eraseBlock ?? { type: "Block", stmts: [] } as BlockNode, s);
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Handler: Extern */
defaultHandlers["Extern"] = (node: ExternNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  s.push('extern ');
  if (node.abi) s.push(`"${node.abi}" `);
  s.push("func ");
  s.push(node.name);
  s.push("(" + (node.params || []).join(", ") + ");");
  s.newline();
  renderTrailingComments(node, s);
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* Fallback generic handler (prints object-ish) */
defaultHandlers["default"] = (node: BaseNode, s: RendererState) => {
  const start = s.pos;
  renderLeadingComments(node, s);
  // best-effort: if node has a `text` field, use it
  if (typeof node.text === "string") {
    s.push(node.text);
    renderTrailingComments(node, s);
    s.finalizeNode(node, s.pos);
    return s.code.slice(start, s.pos);
  }
  // Otherwise, if the node is a simple object representing an expression, serialize in sensible form
  if (node.type === "Object") {
    // object literal
    const keys = Object.keys(node).filter(k => !["type", "id", "loc", "leadingComments", "trailingComments"].includes(k));
    s.push("{");
    if (!s.opts.compact) s.push(" ");
    for (let i = 0; i < keys.length; i++) {
      if (i) s.push(", ");
      const k = keys[i];
      s.push(k + ": ");
      const v = (node as any)[k];
      if (typeof v === "string") s.push(`"${escString(v)}"`);
      else s.push(String(v));
    }
    if (!s.opts.compact) s.push(" ");
    s.push("}");
    renderTrailingComments(node, s);
    s.finalizeNode(node, s.pos);
    return s.code.slice(start, s.pos);
  }
  // As a last resort, JSON stringify
  s.push(JSON.stringify(node));
  s.finalizeNode(node, s.pos);
  return s.code.slice(start, s.pos);
};

/* -------------------------
   renderNode: central dispatcher
   ------------------------- */
export function renderNode(node: BaseNode, state: RendererState): string {
  if (!node) return "";
  // ensure id present if sourceMap emission expected
  if (state.opts.emitSourceMap && !node.id) node.id = genNodeId();
  const startPos = state.pos;
  state.recordNode(node, startPos);
  const handler = (defaultHandlers as any)[node.type] ?? defaultHandlers["default"];
  try {
    const fragment = handler(node, state);
    state.finalizeNode(node, state.pos);
    return fragment;
  } catch (err) {
    // on error: emit a safe comment and JSON dump so rendering continues
    const msg = `/* RENDER ERROR: ${(err as Error).message || String(err)} */`;
    state.push(msg);
    state.finalizeNode(node, state.pos);
    return msg;
  }
}

/* -------------------------
   Public API: renderAst
   ------------------------- */

export function renderAst(ast: InstryxNode | InstryxNode[], opts: RenderOptions = {}): RenderResult {
  const state = new RendererState(opts);
  try {
    if (Array.isArray(ast)) {
      // wrap into Program
      const program: ProgramNode = { type: "Program", body: ast as BaseNode[] };
      renderNode(program as BaseNode, state);
    } else {
      if (ast.type === "Program") {
        renderNode(ast as ProgramNode, state);
      } else {
        // single node -> render as module with one statement
        const program: ProgramNode = { type: "Program", body: [ast as BaseNode] };
        renderNode(program, state);
      }
    }
    const out = state.code;
    if (state.opts.emitSourceMap) {
      return { code: out, sourceMap: state.sourceMap };
    }
    return { code: out };
  } catch (err) {
    // on render failure return JSON fallback
    const fallback = `/* RENDER FAILED: ${(err as Error).message || String(err)} */\n` + JSON.stringify(ast, null, 2);
    return { code: fallback };
  }
}

/* -------------------------
   Export convenience helpers
   ------------------------- */

/**
 * Pretty-print AST with default options.
 */
export function prettyPrint(ast: InstryxNode | InstryxNode[]): string {
  return renderAst(ast, { indent: "  ", compact: false }).code;
}

/**
 * Minified rendering.
 */
export function minify(ast: InstryxNode | InstryxNode[]): string {
  return renderAst(ast, { compact: true }).code;
}

/* -------------------------
   End of ast_renderer.ts
   ------------------------- */

/**
 * ast_renderer.ts
 *
 * Production-ready AST -> Instryx source renderer with optimizations and tooling.
 *
 * Features:
 * - Deterministic pretty-printer (configurable indentation, compact mode).
 * - Built-in optimization pipeline: constant-folding, dead-code-elimination, peephole, inline-small-functions.
 * - Pluggable renderer node handlers and pluggable optimization passes.
 * - Incremental rendering with node-level caching keyed by stable node-hash.
 * - Optional source-map-like node -> [start,end] map.
 * - File I/O helpers: renderToFile, watchAndRender (simple watcher).
 * - CLI helper (programmatic): renderFileCli(argv).
 *
 * No external dependencies; uses Node builtin `crypto` for hashing when available.
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

export type RenderOptions = {
    indent?: string;           // default "  "
    compact?: boolean;         // minimal whitespace (default false)
    maxInlineLength?: number;  // try single-line below this length
    preserveComments?: boolean;
    emitSourceMap?: boolean;
    enableIncremental?: boolean; // enable node-level caching
};

export type RenderResult = {
    code: string;
    sourceMap?: Record<string, { start: number; end: number }>;
    optimized?: boolean;
};

/* -------------------------
   Basic AST typings (permissive)
   ------------------------- */

export interface BaseNode {
    type: string;
    id?: string;
    loc?: { start: number; end: number } | null;
    leadingComments?: string[];
    trailingComments?: string[];
    [k: string]: any;
}

export interface ProgramNode extends BaseNode {
    type: "Program";
    body: BaseNode[];
}

export interface FuncNode extends BaseNode {
    type: "Func";
    name: string;
    params?: string[];
    body?: BaseNode;
    exported?: boolean;
    extern?: boolean;
}

export interface BlockNode extends BaseNode {
    type: "Block";
    stmts: BaseNode[];
}

export interface AssignNode extends BaseNode {
    type: "Assign";
    target: string | BaseNode;
    value: BaseNode;
}

export interface CallNode extends BaseNode {
    type: "Call";
    callee: string | BaseNode;
    args?: BaseNode[];
}

export interface VarNode extends BaseNode {
    type: "Var";
    name: string;
}

export interface LiteralNode extends BaseNode {
    type: "Literal";
    value: string | number | boolean | null | (string | number | boolean | null)[];
}

export interface BinaryNode extends BaseNode {
    type: "Binary";
    op: string;
    left: BaseNode;
    right: BaseNode;
}

export interface UnaryNode extends BaseNode {
    type: "Unary";
    op: string;
    expr: BaseNode;
}

export interface CommentNode extends BaseNode {
    type: "Comment";
    text: string;
    multiline?: boolean;
}

export interface QuarantineNode extends BaseNode {
    type: "Quarantine";
    tryBlock: BlockNode;
    replaceBlock?: BlockNode | null;
    eraseBlock?: BlockNode | null;
}

export interface ExternNode extends BaseNode {
    type: "Extern";
    name: string;
    params?: string[];
    abi?: string;
}

export type InstryxNode =
    | ProgramNode | FuncNode | BlockNode | AssignNode | CallNode | VarNode | LiteralNode
    | BinaryNode | UnaryNode | CommentNode | QuarantineNode | ExternNode | BaseNode;

/* -------------------------
   Utilities: hashing, canonical JSON
   ------------------------- */

function canonicalJson(obj: any): string {
    // deterministic stringify: sort object keys recursively
    return JSON.stringify(obj, function (_k, v) {
        if (v && typeof v === "object" && !Array.isArray(v)) {
            const keys = Object.keys(v).sort();
            const out: any = {};
            for (const k of keys) out[k] = (v as any)[k];
            return out;
        }
        return v;
    }, 0);
}

function sha1(text: string): string {
    try {
        return crypto.createHash("sha1").update(text, "utf8").digest("hex");
    } catch {
        // fallback naive
        let h = 2166136261 >>> 0;
        for (let i = 0; i < text.length; i++) {
            h ^= text.charCodeAt(i);
            h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
        }
        return (h >>> 0).toString(16);
    }
}

/* -------------------------
   Renderer core
   ------------------------- */

type NodeHandler = (node: BaseNode, state: RendererState) => string;

/** RenderState keeps output buffer, indentation, options and incremental cache */
class RendererState {
    opts: Required<RenderOptions>;
    indentLevel: number;
    out: string[];
    pos: number;
    sourceMap: Record<string, { start: number; end: number }>;
    // incremental caches
    nodeCache: Map<string, { fragment: string; start: number; end: number }>;
    constructor(opts: RenderOptions = {}) {
        this.opts = {
            indent: opts.indent ?? "  ",
            compact: !!opts.compact,
            maxInlineLength: opts.maxInlineLength ?? 80,
            preserveComments: !!opts.preserveComments,
            emitSourceMap: !!opts.emitSourceMap,
            enableIncremental: !!opts.enableIncremental,
        };
        this.indentLevel = 0;
        this.out = [];
        this.pos = 0;
        this.sourceMap = {};
        this.nodeCache = new Map();
    }

    push(s: string) {
        this.out.push(s);
        this.pos += s.length;
    }
    pushSpace() {
        if (!this.opts.compact) this.push(" ");
    }
    newline() {
        if (!this.opts.compact) this.push("\n");
    }
    indent() {
        if (!this.opts.compact) this.push(this.opts.indent.repeat(this.indentLevel));
    }

    record(node: BaseNode, start: number) {
        if (!this.opts.emitSourceMap) return;
        const id = node.id ?? genNodeId();
        node.id = id;
        if (!this.sourceMap[id]) this.sourceMap[id] = { start, end: start };
    }
    finalize(node: BaseNode, end: number) {
        if (!this.opts.emitSourceMap) return;
        const id = node.id!;
        const entry = this.sourceMap[id];
        if (entry) entry.end = end;
        else this.sourceMap[id] = { start: 0, end };
    }

    get code() {
        return this.out.join("");
    }
}

/* Node id generator */
let __nodeCounter = 1;
function genNodeId(): string {
    return `n${__nodeCounter++}`;
}

/* -------------------------
   Default node handlers
   ------------------------- */

const handlers: Map<string, NodeHandler> = new Map();

/* Helpers */
function escString(s: string) {
    return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
}

function renderLeadingComments(node: BaseNode, s: RendererState) {
    if (!s.opts.preserveComments) return;
    const arr = node.leadingComments ?? [];
    for (const c of arr) {
        s.indent();
        if (c.includes("\n")) s.push("/* " + c.replace(/\*\//g, "*\\/") + " */");
        else s.push("-- " + c);
        s.newline();
    }
}

function renderTrailingComments(node: BaseNode, s: RendererState) {
    if (!s.opts.preserveComments) return;
    const arr = node.trailingComments ?? [];
    for (const c of arr) {
        s.push(" /* " + c.replace(/\*\//g, "*\\/") + " */");
    }
}

/* Program */
handlers.set("Program", (node: ProgramNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    const body = node.body ?? [];
    for (let i = 0; i < body.length; i++) {
        const child = body[i];
        s.indent();
        renderNode(child, s);
        // append semicolon when appropriate
        if (!s.opts.compact) {
            if (!s.code.endsWith(";") && child.type !== "Func" && child.type !== "Block" && child.type !== "Comment") s.push(";");
        } else {
            if (child.type !== "Func" && child.type !== "Block" && child.type !== "Comment") s.push(";");
        }
        s.newline();
    }
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Func */
handlers.set("Func", (node: FuncNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    if (node.exported) s.push("export ");
    if (node.extern) s.push('extern ');
    s.push("func ");
    s.push(node.name);
    s.push("(" + (node.params ?? []).join(", ") + ")");
    if (node.extern) {
        s.push(";");
        s.newline();
        s.finalize(node, s.pos);
        return s.code.slice(start, s.pos);
    }
    s.push(" ");
    renderNode(node.body ?? { type: "Block", stmts: [] } as BlockNode, s);
    s.finalize(node, s.pos);
    renderTrailingComments(node, s);
    return s.code.slice(start, s.pos);
});

/* Block */
handlers.set("Block", (node: BlockNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push("{");
    if (!s.opts.compact) s.newline();
    s.indentLevel++;
    for (const stmt of node.stmts ?? []) {
        s.indent();
        renderNode(stmt, s);
        if (!s.opts.compact) {
            if (!s.code.endsWith(";") && stmt.type !== "Func" && stmt.type !== "Block" && stmt.type !== "Comment") s.push(";");
        } else {
            if (stmt.type !== "Func" && stmt.type !== "Block" && stmt.type !== "Comment") s.push(";");
        }
        s.newline();
    }
    s.indentLevel--;
    s.indent();
    s.push("}");
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Assign */
handlers.set("Assign", (node: AssignNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    if (typeof node.target === "string") s.push(node.target + " = ");
    else renderNode(node.target as BaseNode, s), s.push(" = ");
    renderNode(node.value, s);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Call */
handlers.set("Call", (node: CallNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    if (typeof node.callee === "string") s.push(node.callee);
    else renderNode(node.callee as BaseNode, s);
    s.push("(");
    const args = node.args ?? [];
    for (let i = 0; i < args.length; i++) {
        if (i) s.push(",");
        if (!s.opts.compact) s.push(" ");
        renderNode(args[i], s);
    }
    if (!s.opts.compact && args.length > 0) s.push(" ");
    s.push(")");
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Var */
handlers.set("Var", (node: VarNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push(node.name);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Literal */
handlers.set("Literal", (node: LiteralNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    const v = node.value;
    if (typeof v === "string") s.push(`"${escString(v)}"`);
    else if (typeof v === "number") s.push(String(v));
    else if (typeof v === "boolean") s.push(v ? "true" : "false");
    else if (v === null) s.push("null");
    else if (Array.isArray(v)) {
        s.push("[");
        for (let i = 0; i < v.length; i++) {
            if (i) s.push(", ");
            const el = v[i];
            if (typeof el === "string") s.push(`"${escString(el)}"`);
            else s.push(String(el));
        }
        s.push("]");
    } else s.push(JSON.stringify(v));
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Binary */
handlers.set("Binary", (node: BinaryNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    renderNode(node.left, s);
    s.push(" " + node.op + " ");
    renderNode(node.right, s);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Unary */
handlers.set("Unary", (node: UnaryNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push(node.op);
    renderNode(node.expr, s);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Comment */
handlers.set("Comment", (node: CommentNode, s: RendererState) => {
    const start = s.pos;
    if (node.multiline) s.push("/* " + (node.text ?? "") + " */");
    else s.push("-- " + (node.text ?? ""));
    s.newline();
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Quarantine */
handlers.set("Quarantine", (node: QuarantineNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push("quarantine try ");
    renderNode(node.tryBlock, s);
    s.push(" replace ");
    renderNode(node.replaceBlock ?? ({ type: "Block", stmts: [] } as BlockNode), s);
    s.push(" erase ");
    renderNode(node.eraseBlock ?? ({ type: "Block", stmts: [] } as BlockNode), s);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Extern */
handlers.set("Extern", (node: ExternNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push('extern ');
    if (node.abi) s.push(`"${node.abi}" `);
    s.push("func ");
    s.push(node.name);
    s.push("(" + (node.params ?? []).join(", ") + ");");
    s.newline();
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* Default fallback */
handlers.set("default", (node: BaseNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    if (typeof (node as any).text === "string") {
        s.push((node as any).text);
    } else {
        s.push(JSON.stringify(node));
    }
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* -------------------------
   Render dispatcher with incremental caching
   ------------------------- */

export function renderNode(node: BaseNode, state: RendererState): string {
    if (!node) return "";
    // incremental cache key computed from canonical JSON of node shape (without id/loc)
    const nodeClone = { ...node };
    delete (nodeClone as any).id;
    delete (nodeClone as any).loc;
    const key = sha1(canonicalJson(nodeClone));

    if (state.opts.enableIncremental && state.nodeCache.has(key)) {
        const cached = state.nodeCache.get(key)!;
        // insert cached fragment and update sourceMap if requested
        const start = state.pos;
        state.push(cached.fragment);
        if (state.opts.emitSourceMap && node.id) state.sourceMap[node.id] = { start, end: start + cached.fragment.length };
        return cached.fragment;
    }

    const startPos = state.pos;
    state.record(node, startPos);
    const handler = handlers.get(node.type) ?? handlers.get("default")!;
    let fragment: string;
    try {
        fragment = handler(node, state);
    } catch (err) {
        const msg = `/* RENDER ERROR: ${(err as Error).message ?? String(err)} */`;
        state.push(msg);
        fragment = msg;
    }
    state.finalize(node, state.pos);
    if (state.opts.enableIncremental) {
        state.nodeCache.set(key, { fragment, start: startPos, end: state.pos });
    }
    return fragment;
}

/* -------------------------
   Optimization passes
   ------------------------- */

/**
 * Constant folding (simple) - folds binary operations with literal operands.
 * Works recursively and returns modified AST.
 */
export function constantFolding(root: InstryxNode): InstryxNode {
    function fold(node: any): any {
        if (!node || typeof node !== "object") return node;
        if (node.type === "Binary") {
            node.left = fold(node.left);
            node.right = fold(node.right);
            const L = node.left, R = node.right;
            if (L?.type === "Literal" && R?.type === "Literal") {
                const a = L.value, b = R.value;
                try {
                    switch (node.op) {
                        case "+": return { type: "Literal", value: (a as any) + (b as any) };
                        case "-": return { type: "Literal", value: (a as any) - (b as any) };
                        case "*": return { type: "Literal", value: (a as any) * (b as any) };
                        case "/": return { type: "Literal", value: (b as any) === 0 ? null : (a as any) / (b as any) };
                        case "==": return { type: "Literal", value: a === b };
                        case "!=": return { type: "Literal", value: a !== b };
                        case ">": return { type: "Literal", value: a > b };
                        case "<": return { type: "Literal", value: a < b };
                        case "&&": return { type: "Literal", value: !!a && !!b };
                        case "||": return { type: "Literal", value: !!a || !!b };
                    }
                } catch {
                    // ignore
                }
            }
            return node;
        } else if (node.type === "Unary") {
            node.expr = fold(node.expr);
            if (node.expr?.type === "Literal") {
                const v = node.expr.value;
                switch (node.op) {
                    case "-": return { type: "Literal", value: -(v as any) };
                    case "!": return { type: "Literal", value: !(v as any) };
                }
            }
            return node;
        } else {
            // traverse children generically
            for (const k of Object.keys(node)) {
                if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
                const v = node[k];
                if (Array.isArray(v)) node[k] = v.map((it) => fold(it));
                else if (v && typeof v === "object") node[k] = fold(v);
            }
            return node;
        }
    }
    return fold(root) as InstryxNode;
}

/**
 * Dead code elimination (conservative):
 * - removes functions not reachable from `main` and not exported/extern
 * - removes top-level assignments whose variables never used (simple)
 */
export function deadCodeElimination(root: InstryxNode): InstryxNode {
    if (!root || (root as any).type !== "Program") return root;
    const prog = root as ProgramNode;
    const functions: Record<string, FuncNode> = {};
    const externals = new Set<string>();
    // collect functions and externs
    for (const n of prog.body) {
        if ((n as any).type === "Func") functions[(n as FuncNode).name] = n as FuncNode;
        if ((n as any).type === "Extern") externals.add((n as ExternNode).name);
    }
    // build call graph
    const callees: Record<string, Set<string>> = {};
    function collectCalls(node: any, curFn?: string) {
        if (!node || typeof node !== "object") return;
        if (node.type === "Call") {
            const callee = typeof node.callee === "string" ? node.callee : null;
            if (curFn && callee) {
                callees[curFn] = callees[curFn] || new Set();
                callees[curFn].add(callee);
            }
        }
        for (const k of Object.keys(node)) {
            if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
            const v = node[k];
            if (Array.isArray(v)) v.forEach((it) => collectCalls(it, curFn));
            else if (v && typeof v === "object") collectCalls(v, curFn);
        }
    }
    for (const name of Object.keys(functions)) {
        collectCalls(functions[name].body, name);
    }
    // find roots: exported/extern + main
    const roots = new Set<string>();
    if (functions["main"]) roots.add("main");
    for (const [n, fn] of Object.entries(functions)) {
        if (fn.exported) roots.add(n);
        if (fn.extern) roots.add(n);
    }
    // reachable traversal
    const reachable = new Set<string>();
    const stack = Array.from(roots);
    while (stack.length) {
        const cur = stack.pop()!;
        if (reachable.has(cur)) continue;
        reachable.add(cur);
        for (const c of Array.from(callees[cur] ?? [])) {
            if (!reachable.has(c)) stack.push(c);
        }
    }
    // filter program body: keep nodes that are Func and reachable or Extern or others
    prog.body = prog.body.filter((n) => {
        if ((n as any).type === "Func") {
            return reachable.has((n as FuncNode).name);
        }
        return true;
    });
    // simple dead-store elimination inside each block: remove assignments where target not used
    function removeDeadStores(block: any) {
        if (!block || typeof block !== "object") return;
        if (block.type === "Block") {
            const uses = new Set<string>();
            for (const s of block.stmts) {
                collectVariables(s, uses);
            }
            const newStmts = [];
            for (const s of block.stmts) {
                if (s.type === "Assign" && typeof s.target === "string") {
                    const tgt = s.target;
                    if (!uses.has(tgt) && isSideEffectFree(s.value)) {
                        continue; // drop
                    }
                }
                newStmts.push(s);
            }
            block.stmts = newStmts;
        }
        for (const k of Object.keys(block)) {
            const v = block[k];
            if (Array.isArray(v)) v.forEach(removeDeadStores);
            else if (v && typeof v === "object") removeDeadStores(v);
        }
    }
    function collectVariables(node: any, set: Set<string>) {
        if (!node || typeof node !== "object") return;
        if (node.type === "Var") set.add(node.name);
        for (const k of Object.keys(node)) {
            if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
            const v = node[k];
            if (Array.isArray(v)) v.forEach((it) => collectVariables(it, set));
            else if (v && typeof v === "object") collectVariables(v, set);
        }
    }
    function isSideEffectFree(n: any): boolean {
        if (!n) return true;
        if (n.type === "Call") return false;
        if (n.type === "Quarantine") return false;
        // literals, var, binary/unary evaluated safe
        for (const k of Object.keys(n)) {
            const v = n[k];
            if (Array.isArray(v)) { if (v.some((it) => !isSideEffectFree(it))) return false; }
            else if (v && typeof v === "object") { if (!isSideEffectFree(v)) return false; }
        }
        return true;
    }
    // apply to remaining bodies
    for (const n of prog.body) {
        if (n.type === "Func") removeDeadStores(n.body);
    }
    return prog as InstryxNode;
}

/**
 * Peephole optimizations (simple rewrites):
 * - x + 0 -> x
 * - x * 1 -> x
 * - x - 0 -> x
 * - double negation
 */
export function peepholeOptimize(root: InstryxNode): InstryxNode {
    function visit(n: any): any {
        if (!n || typeof n !== "object") return n;
        if (n.type === "Binary") {
            n.left = visit(n.left);
            n.right = visit(n.right);
            const L = n.left, R = n.right;
            if (n.op === "+" && R?.type === "Literal" && R.value === 0) return L;
            if (n.op === "+" && L?.type === "Literal" && L.value === 0) return R;
            if (n.op === "*" && R?.type === "Literal" && R.value === 1) return L;
            if (n.op === "*" && L?.type === "Literal" && L.value === 1) return R;
            if (n.op === "-" && R?.type === "Literal" && R.value === 0) return L;
        } else if (n.type === "Unary") {
            n.expr = visit(n.expr);
            if (n.op === "-" && n.expr?.type === "Unary" && n.expr.op === "-") return n.expr.expr;
        } else {
            for (const k of Object.keys(n)) {
                if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
                const v = n[k];
                if (Array.isArray(v)) n[k] = v.map(visit);
                else if (v && typeof v === "object") n[k] = visit(v);
            }
        }
        return n;
    }
    return visit(root) as InstryxNode;
}

/**
 * Inline small functions: replace call sites to tiny functions with their bodies.
 * Conservative: only zero-arg or simple param matching, small body length threshold.
 */
export function inlineSmallFunctions(root: InstryxNode, maxSize = 150): InstryxNode {
    if (!root || (root as any).type !== "Program") return root;
    const prog = root as ProgramNode;
    const functions: Record<string, FuncNode> = {};
    for (const n of prog.body) if (n.type === "Func") functions[n.name] = n as FuncNode;
    const smallFns = new Set<string>();
    for (const [name, fn] of Object.entries(functions)) {
        const bodyText = canonicalJson(fn.body ?? {});
        if (bodyText.length <= maxSize) smallFns.add(name);
    }
    function cloneDeep(obj: any) { return JSON.parse(JSON.stringify(obj)); }
    function visit(node: any): any {
        if (!node || typeof node !== "object") return node;
        if (node.type === "Call" && typeof node.callee === "string" && smallFns.has(node.callee)) {
            const fn = functions[node.callee];
            if (!fn) return node;
            // map params to args (simple)
            const params = fn.params ?? [];
            const args = node.args ?? [];
            const mapping: Record<string, any> = {};
            for (let i = 0; i < Math.min(params.length, args.length); i++) mapping[params[i]] = args[i];
            // clone function body and replace param occurrences (naive)
            const bodyClone = cloneDeep(fn.body);
            function replaceVars(n: any) {
                if (!n || typeof n !== "object") return n;
                if (n.type === "Var" && typeof n.name === "string" && mapping[n.name]) return mapping[n.name];
                for (const k of Object.keys(n)) {
                    if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
                    const v = n[k];
                    if (Array.isArray(v)) n[k] = v.map(replaceVars);
                    else if (v && typeof v === "object") n[k] = replaceVars(v);
                }
                return n;
            }
            const inlined = replaceVars(bodyClone);
            // if inlined is block, return as-is (may need wrapping)
            return inlined;
        }
        for (const k of Object.keys(node)) {
            if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
            const v = node[k];
            if (Array.isArray(v)) node[k] = v.map(visit);
            else if (v && typeof v === "object") node[k] = visit(v);
        }
        return node;
    }
    return visit(prog) as InstryxNode;
}

/* High-level optimizeAst: runs registered passes */
type OptPass = (ast: InstryxNode) => InstryxNode;
const optPasses: OptPass[] = [constantFolding, peepholeOptimize, deadCodeElimination, inlineSmallFunctions];

/**
 * Register an optimization pass that runs in order
 */
export function registerOptimizationPass(pass: OptPass) {
    optPasses.push(pass);
}

/**
 * Optimize AST with configured passes.
 */
export function optimizeAst(ast: InstryxNode, passes?: OptPass[]): InstryxNode {
    let cur = ast;
    const pipeline = passes ?? optPasses;
    for (const p of pipeline) {
        try {
            cur = p(cur);
        } catch (err) {
            // non-fatal: log and continue
            console.warn("optimizeAst pass failed:", (err as Error).message ?? err);
        }
    }
    return cur;
}

/* -------------------------
   Public rendering API
   ------------------------- */

export function renderAst(ast: InstryxNode | InstryxNode[], opts: RenderOptions = {}): RenderResult {
    const state = new RendererState(opts);
    try {
        let program: ProgramNode;
        if (Array.isArray(ast)) program = { type: "Program", body: ast as BaseNode[] };
        else if ((ast as any).type === "Program") program = ast as ProgramNode;
        else program = { type: "Program", body: [ast as BaseNode] };
        // Optional optimize before render
        if ((opts as any).optimize) program = optimizeAst(program) as ProgramNode;
        renderNode(program, state);
        const code = state.code;
        if (state.opts.emitSourceMap) {
            return { code, sourceMap: state.sourceMap, optimized: !!(opts as any).optimize };
        }
        return { code, optimized: !!(opts as any).optimize };
    } catch (err) {
        const fallback = "/* RENDER FAILED: " + ((err as Error).message ?? String(err)) + " */\n" + JSON.stringify(ast, null, 2);
        return { code: fallback };
    }
}

/* Convenience pretty/minify */
export function prettyPrint(ast: InstryxNode | InstryxNode[]): string {
    return renderAst(ast, { indent: "  ", compact: false }).code;
}
export function minify(ast: InstryxNode | InstryxNode[]): string {
    return renderAst(ast, { compact: true }).code;
}

/* -------------------------
   Extensibility: register node handler
   ------------------------- */

/**
 * Register a custom node handler for a given node type.
 * Handler must produce string and should call renderNode for subnodes.
 */
export function registerNodeHandler(type: string, handler: NodeHandler) {
    handlers.set(type, handler);
}

/* -------------------------
   File helpers & CLI
   ------------------------- */

export async function renderToFile(inPath: string, outPath?: string, options: RenderOptions & { optimize?: boolean } = {}) {
    const src = await fs.promises.readFile(inPath, "utf8");
    // Expect input is AST JSON for safety (many tools will call render from AST)
    let ast: any;
    try {
        ast = JSON.parse(src);
    } catch {
        throw new Error("renderToFile expects input file to be JSON AST");
    }
    const result = renderAst(ast, { ...options, emitSourceMap: options.emitSourceMap ?? false, enableIncremental: true });
    const dest = outPath ?? inPath.replace(/\.json$/, ".ix");
    await fs.promises.writeFile(dest, result.code, "utf8");
    if (options.emitSourceMap && result.sourceMap) {
        await fs.promises.writeFile(dest + ".map.json", JSON.stringify(result.sourceMap, null, 2), "utf8");
    }
    return dest;
}

/**
 * Watch an AST JSON file for changes and re-render.
 * Very small convenience utility for development.
 */
export function watchAndRender(inPath: string, outPath?: string, options: RenderOptions & { optimize?: boolean } = {}) {
    let timer: NodeJS.Timeout | null = null;
    const renderOnce = async () => {
        try {
            const dest = await renderToFile(inPath, outPath, options);
            console.log(`Rendered ${inPath} -> ${dest}`);
        } catch (err) {
            console.error("watchAndRender failed:", (err as Error).message ?? err);
        }
    };
    fs.watch(inPath, () => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => renderOnce().catch(console.error), 120);
    });
    // initial
    renderOnce().catch(console.error);
}

/* -------------------------
   Simple CLI programmatic entry (for scripts)
   ------------------------- */
export async function renderFileCli(argv: string[]) {
    if (!argv || argv.length < 1) {
        console.error("Usage: renderFileCli <ast.json> [out.ix] [--optimize] [--compact] [--sourcemap]");
        return 2;
    }
    const inPath = argv[0];
    const outPath = argv[1];
    const opts: any = {
        optimize: argv.includes("--optimize"),
        compact: argv.includes("--compact"),
        emitSourceMap: argv.includes("--sourcemap"),
        enableIncremental: true,
    };
    try {
        const dest = await renderToFile(inPath, outPath, opts);
        console.log("Wrote:", dest);
        return 0;
    } catch (err) {
        console.error("renderFileCli error:", (err as Error).message ?? err);
        return 1;
    }
}

/* -------------------------
   Exports
   ------------------------- */
export {
    RendererState,
    renderNode,
    handlers as rendererHandlers,
    dialectOptimizations as _internalOptimizations,
};

/* internal helper list (exposed for testing) */
export const dialectOptimizations = {
    constantFolding,
    peepholeOptimize,
    deadCodeElimination,
    inlineSmallFunctions,
};

/**
 * ast_renderer.ts
 *
 * Production-grade AST -> Instryx source renderer, optimizer and tooling.
 *
 * Major additions over the basic renderer:
 * - Extended optimization pipeline (SSA conversion, CFG simplify, interprocedural
 *   constant propagation, function specialization, profile-guided inlining hooks,
 *   loop unrolling, vectorize hints, dead-code-elim, peephole, constant folding).
 * - Pluggable optimization pass registry and pass ordering controls.
 * - Incremental/disk-backed node cache for fast repeated renders.
 * - Source-map emission (node id -> range).
 * - Verifier, linter and quick-fix suggestions.
 * - CLI helpers: renderFileCli, renderAndOptimizeToFile, watchAndAutoOptimize.
 * - Exportable optimization report and AST snapshotting utilities.
 *
 * No external (npm) dependencies; only Node stdlib is used.
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

export type RenderOptions = {
    indent?: string;
    compact?: boolean;
    maxInlineLength?: number;
    preserveComments?: boolean;
    emitSourceMap?: boolean;
    enableIncremental?: boolean;
    optimize?: boolean;
    optimizePasses?: string[]; // names or ids of passes to run (default pipeline otherwise)
    profile?: Record<string, number>; // function hotness map for PGO-aware passes
};

export type RenderResult = {
    code: string;
    sourceMap?: Record<string, { start: number; end: number }>;
    optimized?: boolean;
    report?: OptimizationReport;
};

export type OptimizationReport = {
    ranPasses: string[];
    timings: Record<string, number>;
    changes: string[]; // human-readable summary of key changes
    astHashBefore?: string;
    astHashAfter?: string;
};

let __internalNodeId = 1;
function genNodeId(): string {
    return `n${__internalNodeId++}`;
}

/* -------------------------
   AST types (permissive)
   ------------------------- */

export interface BaseNode {
    type: string;
    id?: string;
    loc?: { start: number; end: number } | null;
    leadingComments?: string[];
    trailingComments?: string[];
    [k: string]: any;
}

export interface ProgramNode extends BaseNode {
    type: "Program";
    body: BaseNode[];
}

export interface FuncNode extends BaseNode {
    type: "Func";
    name: string;
    params?: string[];
    body?: BaseNode;
    exported?: boolean;
    extern?: boolean;
    annotations?: Record<string, any>;
}

export interface BlockNode extends BaseNode {
    type: "Block";
    stmts: BaseNode[];
}

export interface AssignNode extends BaseNode {
    type: "Assign";
    target: string | BaseNode;
    value: BaseNode;
}

export interface CallNode extends BaseNode {
    type: "Call";
    callee: string | BaseNode;
    args?: BaseNode[];
}

export interface VarNode extends BaseNode {
    type: "Var";
    name: string;
}

export interface LiteralNode extends BaseNode {
    type: "Literal";
    value: string | number | boolean | null | (string | number | boolean | null)[];
}

export interface BinaryNode extends BaseNode {
    type: "Binary";
    op: string;
    left: BaseNode;
    right: BaseNode;
}

export interface UnaryNode extends BaseNode {
    type: "Unary";
    op: string;
    expr: BaseNode;
}

export interface CommentNode extends BaseNode {
    type: "Comment";
    text: string;
    multiline?: boolean;
}

export interface QuarantineNode extends BaseNode {
    type: "Quarantine";
    tryBlock: BlockNode;
    replaceBlock?: BlockNode | null;
    eraseBlock?: BlockNode | null;
}

export interface ExternNode extends BaseNode {
    type: "Extern";
    name: string;
    params?: string[];
    abi?: string;
}

export type InstryxNode =
    | ProgramNode | FuncNode | BlockNode | AssignNode | CallNode | VarNode | LiteralNode
    | BinaryNode | UnaryNode | CommentNode | QuarantineNode | ExternNode | BaseNode;

/* -------------------------
   Utilities
   ------------------------- */

function canonicalJson(obj: any): string {
    return JSON.stringify(obj, function (_k, v) {
        if (v && typeof v === "object" && !Array.isArray(v)) {
            const keys = Object.keys(v).sort();
            const out: any = {};
            for (const k of keys) out[k] = (v as any)[k];
            return out;
        }
        return v;
    }, 0);
}

function sha1(text: string) {
    try {
        return crypto.createHash("sha1").update(text, "utf8").digest("hex");
    } catch {
        let h = 2166136261 >>> 0;
        for (let i = 0; i < text.length; i++) {
            h ^= text.charCodeAt(i);
            h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
        }
        return (h >>> 0).toString(16);
    }
}

function deepClone<T>(v: T): T { return JSON.parse(JSON.stringify(v)); }

/* -------------------------
   Renderer core
   ------------------------- */

type NodeHandler = (node: BaseNode, state: RendererState) => string;

class RendererState {
    opts: Required<RenderOptions>;
    indentLevel: number;
    out: string[];
    pos: number;
    sourceMap: Record<string, { start: number; end: number }>;
    nodeCache: Map<string, { fragment: string; start: number; end: number }>;
    constructor(opts: RenderOptions = {}) {
        this.opts = {
            indent: opts.indent ?? "  ",
            compact: !!opts.compact,
            maxInlineLength: opts.maxInlineLength ?? 80,
            preserveComments: !!opts.preserveComments,
            emitSourceMap: !!opts.emitSourceMap,
            enableIncremental: !!opts.enableIncremental,
            optimize: !!opts.optimize,
            optimizePasses: (opts as any).optimizePasses ?? [],
            profile: (opts as any).profile ?? {},
        } as Required<RenderOptions>;
        this.indentLevel = 0;
        this.out = [];
        this.pos = 0;
        this.sourceMap = {};
        this.nodeCache = new Map();
    }

    push(s: string) {
        this.out.push(s);
        this.pos += s.length;
    }
    pushSpace() { if (!this.opts.compact) this.push(" "); }
    newline() { if (!this.opts.compact) this.push("\n"); }
    indent() { if (!this.opts.compact) this.push(this.opts.indent.repeat(this.indentLevel)); }

    record(node: BaseNode, start: number) {
        if (!this.opts.emitSourceMap) return;
        const id = node.id ?? genNodeId();
        node.id = id;
        if (!this.sourceMap[id]) this.sourceMap[id] = { start, end: start };
    }
    finalize(node: BaseNode, end: number) {
        if (!this.opts.emitSourceMap) return;
        const id = node.id!;
        const e = this.sourceMap[id];
        if (e) e.end = end; else this.sourceMap[id] = { start: 0, end };
    }

    get code() { return this.out.join(""); }
}

/* -------------------------
   Handler registry & helpers
   ------------------------- */

const handlers: Map<string, NodeHandler> = new Map();

function escString(s: string) {
    return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
}

function renderLeadingComments(node: BaseNode, s: RendererState) {
    if (!s.opts.preserveComments) return;
    const arr = node.leadingComments ?? [];
    for (const c of arr) {
        s.indent();
        if (c.includes("\n")) s.push("/* " + c.replace(/\*\//g, "*\\/") + " */");
        else s.push("-- " + c);
        s.newline();
    }
}

function renderTrailingComments(node: BaseNode, s: RendererState) {
    if (!s.opts.preserveComments) return;
    const arr = node.trailingComments ?? [];
    for (const c of arr) {
        s.push(" /* " + c.replace(/\*\//g, "*\\/") + " */");
    }
}

/* -------------------------
   Default Node Handlers
   (kept compact but clear)
   ------------------------- */

handlers.set("Program", (node: ProgramNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    const body = node.body ?? [];
    for (let i = 0; i < body.length; i++) {
        const child = body[i];
        s.indent();
        renderNode(child, s);
        if (!s.opts.compact) {
            if (!s.code.endsWith(";") && child.type !== "Func" && child.type !== "Block" && child.type !== "Comment") s.push(";");
        } else {
            if (child.type !== "Func" && child.type !== "Block" && child.type !== "Comment") s.push(";");
        }
        s.newline();
    }
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Func", (node: FuncNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    if (node.exported) s.push("export ");
    if (node.extern) s.push('extern ');
    s.push("func ");
    s.push(node.name);
    s.push("(" + (node.params ?? []).join(", ") + ")");
    if (node.extern) {
        s.push(";");
        s.newline();
        s.finalize(node, s.pos);
        return s.code.slice(start, s.pos);
    }
    s.push(" ");
    renderNode(node.body ?? { type: "Block", stmts: [] } as BlockNode, s);
    s.finalize(node, s.pos);
    renderTrailingComments(node, s);
    return s.code.slice(start, s.pos);
});

handlers.set("Block", (node: BlockNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push("{");
    if (!s.opts.compact) s.newline();
    s.indentLevel++;
    for (const stmt of node.stmts ?? []) {
        s.indent();
        renderNode(stmt, s);
        if (!s.opts.compact) {
            if (!s.code.endsWith(";") && stmt.type !== "Func" && stmt.type !== "Block" && stmt.type !== "Comment") s.push(";");
        } else {
            if (stmt.type !== "Func" && stmt.type !== "Block" && stmt.type !== "Comment") s.push(";");
        }
        s.newline();
    }
    s.indentLevel--;
    s.indent();
    s.push("}");
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Assign", (node: AssignNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    if (typeof node.target === "string") s.push(node.target + " = ");
    else { renderNode(node.target as BaseNode, s); s.push(" = "); }
    renderNode(node.value, s);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Call", (node: CallNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    if (typeof node.callee === "string") s.push(node.callee);
    else renderNode(node.callee as BaseNode, s);
    s.push("(");
    const args = node.args ?? [];
    for (let i = 0; i < args.length; i++) {
        if (i) s.push(",");
        if (!s.opts.compact) s.push(" ");
        renderNode(args[i], s);
    }
    if (!s.opts.compact && args.length > 0) s.push(" ");
    s.push(")");
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Var", (node: VarNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push(node.name);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Literal", (node: LiteralNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    const v = node.value;
    if (typeof v === "string") s.push(`"${escString(v)}"`);
    else if (typeof v === "number") s.push(String(v));
    else if (typeof v === "boolean") s.push(v ? "true" : "false");
    else if (v === null) s.push("null");
    else if (Array.isArray(v)) {
        s.push("[");
        for (let i = 0; i < v.length; i++) {
            if (i) s.push(", ");
            const el = v[i];
            if (typeof el === "string") s.push(`"${escString(el)}"`);
            else s.push(String(el));
        }
        s.push("]");
    } else s.push(JSON.stringify(v));
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Binary", (node: BinaryNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    renderNode(node.left, s);
    s.push(" " + node.op + " ");
    renderNode(node.right, s);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Unary", (node: UnaryNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push(node.op);
    renderNode(node.expr, s);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Comment", (node: CommentNode, s: RendererState) => {
    const start = s.pos;
    if (node.multiline) s.push("/* " + (node.text ?? "") + " */");
    else s.push("-- " + (node.text ?? ""));
    s.newline();
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Quarantine", (node: QuarantineNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push("quarantine try ");
    renderNode(node.tryBlock, s);
    s.push(" replace ");
    renderNode(node.replaceBlock ?? ({ type: "Block", stmts: [] } as BlockNode), s);
    s.push(" erase ");
    renderNode(node.eraseBlock ?? ({ type: "Block", stmts: [] } as BlockNode), s);
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("Extern", (node: ExternNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    s.push('extern ');
    if (node.abi) s.push(`"${node.abi}" `);
    s.push("func ");
    s.push(node.name);
    s.push("(" + (node.params ?? []).join(", ") + ");");
    s.newline();
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

handlers.set("default", (node: BaseNode, s: RendererState) => {
    const start = s.pos;
    renderLeadingComments(node, s);
    if (typeof (node as any).text === "string") s.push((node as any).text);
    else s.push(JSON.stringify(node));
    renderTrailingComments(node, s);
    s.finalize(node, s.pos);
    return s.code.slice(start, s.pos);
});

/* -------------------------
   Render dispatcher w/ incremental caching
   ------------------------- */

export function renderNode(node: BaseNode, state: RendererState): string {
    if (!node) return "";
    // incremental cache key computed from canonical JSON of node shape (without id/loc)
    const nodeClone = { ...node };
    delete (nodeClone as any).id;
    delete (nodeClone as any).loc;
    const key = sha1(canonicalJson(nodeClone));

    if (state.opts.enableIncremental && state.nodeCache.has(key)) {
        const cached = state.nodeCache.get(key)!;
        const start = state.pos;
        state.push(cached.fragment);
        if (state.opts.emitSourceMap && node.id) state.sourceMap[node.id] = { start, end: start + cached.fragment.length };
        return cached.fragment;
    }

    const startPos = state.pos;
    state.record(node, startPos);
    const handler = handlers.get(node.type) ?? handlers.get("default")!;
    let fragment: string;
    try {
        fragment = handler(node, state);
    } catch (err) {
        const msg = `/* RENDER ERROR: ${(err as Error).message || String(err)} */`;
        state.push(msg);
        fragment = msg;
    }
    state.finalize(node, state.pos);
    if (state.opts.enableIncremental) {
        state.nodeCache.set(key, { fragment, start: startPos, end: state.pos });
    }
    return fragment;
}

/* -------------------------
   Optimization passes (extended)
   ------------------------- */

/**
 * Constant folding
 */
export function constantFolding(root: InstryxNode): InstryxNode {
    function fold(node: any): any {
        if (!node || typeof node !== "object") return node;
        if (node.type === "Binary") {
            node.left = fold(node.left);
            node.right = fold(node.right);
            const L = node.left, R = node.right;
            if (L?.type === "Literal" && R?.type === "Literal") {
                const a = L.value, b = R.value;
                try {
                    switch (node.op) {
                        case "+": return { type: "Literal", value: (a as any) + (b as any) };
                        case "-": return { type: "Literal", value: (a as any) - (b as any) };
                        case "*": return { type: "Literal", value: (a as any) * (b as any) };
                        case "/": return { type: "Literal", value: (b as any) === 0 ? null : (a as any) / (b as any) };
                        case "==": return { type: "Literal", value: a === b };
                        case "!=": return { type: "Literal", value: a !== b };
                        case ">": return { type: "Literal", value: a > b };
                        case "<": return { type: "Literal", value: a < b };
                        case "&&": return { type: "Literal", value: !!a && !!b };
                        case "||": return { type: "Literal", value: !!a || !!b };
                    }
                } catch { }
            }
            return node;
        } else if (node.type === "Unary") {
            node.expr = fold(node.expr);
            if (node.expr?.type === "Literal") {
                const v = node.expr.value;
                switch (node.op) {
                    case "-": return { type: "Literal", value: -(v as any) };
                    case "!": return { type: "Literal", value: !(v as any) };
                }
            }
            return node;
        } else {
            for (const k of Object.keys(node)) {
                if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
                const v = node[k];
                if (Array.isArray(v)) node[k] = v.map((it) => fold(it));
                else if (v && typeof v === "object") node[k] = fold(v);
            }
            return node;
        }
    }
    return fold(root) as InstryxNode;
}

/**
 * SSA-like renaming (lightweight) - useful for reducing variable collisions before passes.
 */
export function ssaConvert(root: InstryxNode): InstryxNode {
    let counter = 0;
    const mappingStack: Array<Record<string, string>> = [];

    function pushScope() { mappingStack.push({}); }
    function popScope() { mappingStack.pop(); }
    function setName(orig: string) {
        const m = mappingStack[mappingStack.length - 1];
        const name = `${orig}__ssa${counter++}`;
        m[orig] = name;
        return name;
    }
    function lookupName(orig: string) {
        for (let i = mappingStack.length - 1; i >= 0; i--) {
            if (mappingStack[i][orig]) return mapping[i][orig];
        }
        return orig;
    }

    function visit(n: any): any {
        if (!n || typeof n !== "object") return n;
        if (n.type === "Block") {
            pushScope();
            n.stmts = n.stmts.map(visit);
            popScope();
            return n;
        }
        if (n.type === "Assign" && typeof n.target === "string") {
            const newName = setName(n.target);
            n.target = newName;
            n.value = visit(n.value);
            return n;
        }
        if (n.type === "Var") {
            if (typeof n.name === "string") {
                const ln = lookupName(n.name);
                n.name = ln;
            }
            return n;
        }
        for (const k of Object.keys(n)) {
            if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
            const v = n[k];
            if (Array.isArray(v)) n[k] = v.map(visit);
            else if (v && typeof v === "object") n[k] = visit(v);
        }
        return n;
    }

    return visit(root) as InstryxNode;
}

/**
 * CFG simplify - flatten nested blocks and remove empty statements
 */
export function cfgSimplify(root: InstryxNode): InstryxNode {
    function visit(n: any): any {
        if (!n || typeof n !== "object") return n;
        if (n.type === "Block") {
            const out: any[] = [];
            for (const s of (n.stmts ?? [])) {
                const ss = visit(s);
                if (ss == null) continue;
                if (ss.type === "Block") out.push(...(ss.stmts ?? []));
                else out.push(ss);
            }
            n.stmts = out;
            return n;
        }
        for (const k of Object.keys(n)) {
            if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
            const v = n[k];
            if (Array.isArray(v)) n[k] = v.map(visit).filter(Boolean);
            else if (v && typeof v === "object") n[k] = visit(v);
        }
        return n;
    }
    return visit(root) as InstryxNode;
}

/**
 * Interprocedural constant propagation (very conservative):
 * - Propagates function-level known constants for zero-arg calls returning literals.
 */
export function interproceduralConstantPropagation(root: InstryxNode): InstryxNode {
    if (!root || (root as any).type !== "Program") return root;
    const prog = root as ProgramNode;
    const constReturnFns = new Map<string, any>();
    // collect functions that return literal unconditionally (very naive)
    for (const n of prog.body) {
        if (n.type === "Func") {
            const body = (n as FuncNode).body;
            if (body && (body as any).type === "Block") {
                const stmts = (body as BlockNode).stmts;
                if (stmts.length === 1 && stmts[0].type === "Return" && (stmts[0] as ReturnNode).value?.type === "Literal") {
                    constReturnFns.set((n as FuncNode).name, (stmts[0] as ReturnNode).value.value);
                }
            }
        }
    }
    // replace zero-arg calls to these functions with literal
    function visit(n: any): any {
        if (!n || typeof n !== "object") return n;
        if (n.type === "Call" && typeof n.callee === "string" && (n.args ?? []).length === 0) {
            if (constReturnFns.has(n.callee)) return { type: "Literal", value: constReturnFns.get(n.callee) };
        }
        for (const k of Object.keys(n)) {
            if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
            const v = n[k];
            if (Array.isArray(v)) n[k] = v.map(visit);
            else if (v && typeof v === "object") n[k] = visit(v);
        }
        return n;
    }
    return visit(prog) as InstryxNode;
}

/**
 * Function specialization for constant args (simple).
 */
export function functionSpecialization(root: InstryxNode, maxSpecialSize = 200): InstryxNode {
    if (!root || (root as any).type !== "Program") return root;
    const prog = root as ProgramNode;
    const functions: Record<string, FuncNode> = {};
    for (const n of prog.body) if (n.type === "Func") functions[(n as FuncNode).name] = n as FuncNode;
    const newFuncs: Record<string, FuncNode> = {};
    for (const fname of Object.keys(functions)) newFuncs[fname] = functions[fname];

    function visit(n: any): any {
        if (!n || typeof n !== "object") return n;
        if (n.type === "Call" && typeof n.callee === "string") {
            const fn = functions[n.callee];
            if (!fn) return n;
            const args = n.args ?? [];
            // if all args are literal and function small, create specialized function
            if (args.length > 0 && args.every((a: any) => a?.type === "Literal")) {
                const bodyText = canonicalJson(fn.body ?? {});
                if (bodyText.length <= maxSpecialSize) {
                    const key = sha1(canonicalJson(args));
                    const specName = `${fn.name}__spec__${key.slice(0, 8)}`;
                    if (!newFuncs[specName]) {
                        // clone fn and replace params with literal args
                        const clone = deepClone(fn) as FuncNode;
                        const params = clone.params ?? [];
                        const mapping: Record<string, any> = {};
                        for (let i = 0; i < Math.min(params.length, args.length); i++) mapping[params[i]] = args[i];
                        function replaceVars(n2: any): any {
                            if (!n2 || typeof n2 !== "object") return n2;
                            if (n2.type === "Var" && typeof n2.name === "string" && mapping[n2.name]) return mapping[n2.name];
                            for (const k of Object.keys(n2)) {
                                const v = n2[k];
                                if (Array.isArray(v)) n2[k] = v.map(replaceVars);
                                else if (v && typeof v === "object") n2[k] = replaceVars(v);
                            }
                            return n2;
                        }
                        clone.name = specName;
                        clone.params = [];
                        clone.body = replaceVars(clone.body);
                        newFuncs[specName] = clone;
                    }
                    return { type: "Call", callee: specName, args: [] } as CallNode;
                }
            }
        }
        for (const k of Object.keys(n)) {
            if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
            const v = n[k];
            if (Array.isArray(v)) n[k] = v.map(visit);
            else if (v && typeof v === "object") n[k] = visit(v);
        }
        return n;
    }

    const transformed = visit(prog) as ProgramNode;
    // attach new specialized functions at end
    transformed.body = transformed.body.concat(Object.values(newFuncs).filter(f => !functions[f.name]));
    return transformed as InstryxNode;
}

/**
 * Loop unrolling (simple heuristic for small constant-count loops).
 */
export function loopUnroll(root: InstryxNode, maxUnroll = 8): InstryxNode {
    function visit(n: any): any {
        if (!n || typeof n !== "object") return n;
        if (n.type === "For" && typeof n.to === "object" && n.to.type === "Literal" && typeof n.to.value === "number") {
            const count = n.to.value;
            if (count > 0 && count <= maxUnroll && n.body) {
                const stmts: any[] = [];
                for (let i = 0; i < count; i++) {
                    // deep clone body and replace index var if present
                    const clone = deepClone(n.body);
                    if (n.index) {
                        function repl(node2: any) {
                            if (!node2 || typeof node2 !== "object") return node2;
                            if (node2.type === "Var" && node2.name === n.index) return { type: "Literal", value: i };
                            for (const k of Object.keys(node2)) {
                                node2[k] = repl(node2[k]);
                            }
                            return node2;
                        }
                        repl(clone);
                    }
                    if (clone.type === "Block") stmts.push(...clone.stmts);
                    else stmts.push(clone);
                }
                return { type: "Block", stmts } as BlockNode;
            }
        }
        for (const k of Object.keys(n)) {
            const v = n[k];
            if (Array.isArray(v)) n[k] = v.map(visit);
            else if (v && typeof v === "object") n[k] = visit(v);
        }
        return n;
    }
    return visit(root) as InstryxNode;
}

/**
 * Peephole optimization
 */
export function peepholeOptimize(root: InstryxNode): InstryxNode {
    function visit(n: any): any {
        if (!n || typeof n !== "object") return n;
        if (n.type === "Binary") {
            n.left = visit(n.left);
            n.right = visit(n.right);
            if (n.op === "+" && n.right?.type === "Literal" && n.right.value === 0) return n.left;
            if (n.op === "+" && n.left?.type === "Literal" && n.left.value === 0) return n.right;
            if (n.op === "*" && n.right?.type === "Literal" && n.right.value === 1) return n.left;
            if (n.op === "*" && n.left?.type === "Literal" && n.left.value === 1) return n.right;
            if (n.op === "-" && n.right?.type === "Literal" && n.right.value === 0) return n.left;
        } else if (n.type === "Unary") {
            n.expr = visit(n.expr);
            if (n.op === "-" && n.expr?.type === "Unary" && n.expr.op === "-") return n.expr.expr;
        } else {
            for (const k of Object.keys(n)) {
                const v = n[k];
                if (Array.isArray(v)) n[k] = v.map(visit);
                else if (v && typeof v === "object") n[k] = visit(v);
            }
        }
        return n;
    }
    return visit(root) as InstryxNode;
}

/**
 * Dead code elimination (conservative)
 */
export function deadCodeElimination(root: InstryxNode): InstryxNode {
    if (!root || (root as any).type !== "Program") return root;
    const prog = root as ProgramNode;
    const functions: Record<string, FuncNode> = {};
    for (const n of prog.body) if (n.type === "Func") functions[(n as FuncNode).name] = n as FuncNode;
    const callees: Record<string, Set<string>> = {};
    function collectCalls(node: any, curFn?: string) {
        if (!node || typeof node !== "object") return;
        if (node.type === "Call") {
            const callee = typeof node.callee === "string" ? node.callee : null;
            if (curFn && callee) {
                callees[curFn] = callees[curFn] || new Set();
                callees[curFn].add(callee);
            }
        }
        for (const k of Object.keys(node)) {
            const v = node[k];
            if (Array.isArray(v)) v.forEach((it) => collectCalls(it, curFn));
            else if (v && typeof v === "object") collectCalls(v, curFn);
        }
    }
    for (const name of Object.keys(functions)) collectCalls(functions[name].body, name);
    const roots = new Set<string>();
    if (functions["main"]) roots.add("main");
    for (const [n, fn] of Object.entries(functions)) {
        if (fn.exported) roots.add(n);
        if (fn.extern) roots.add(n);
    }
    const reachable = new Set<string>();
    const stack = Array.from(roots);
    while (stack.length) {
        const cur = stack.pop()!;
        if (reachable.has(cur)) continue;
        reachable.add(cur);
        for (const c of Array.from(callees[cur] ?? [])) if (!reachable.has(c)) stack.push(c);
    }
    prog.body = prog.body.filter((n) => {
        if ((n as any).type === "Func") return reachable.has((n as FuncNode).name);
        return true;
    });
    // local dead store elimination
    function removeDeadStores(block: any) {
        if (!block || typeof block !== "object") return;
        if (block.type === "Block") {
            const uses = new Set<string>();
            for (const s of block.stmts) collectVars(s, uses);
            const newStmts: any[] = [];
            for (const s of block.stmts) {
                if (s.type === "Assign" && typeof s.target === "string") {
                    const tgt = s.target;
                    if (!uses.has(tgt) && sideEffectFree(s.value)) continue;
                }
                newStmts.push(s);
            }
            block.stmts = newStmts;
        }
        for (const k of Object.keys(block)) {
            const v = block[k];
            if (Array.isArray(v)) v.forEach(removeDeadStores);
            else if (v && typeof v === "object") removeDeadStores(v);
        }
    }
    function collectVars(node: any, set: Set<string>) {
        if (!node || typeof node !== "object") return;
        if (node.type === "Var") set.add(node.name);
        for (const k of Object.keys(node)) {
            const v = node[k];
            if (Array.isArray(v)) v.forEach((it) => collectVars(it, set));
            else if (v && typeof v === "object") collectVars(v, set);
        }
    }
    function sideEffectFree(n: any): boolean {
        if (!n) return true;
        if (n.type === "Call") return false;
        if (n.type === "Quarantine") return false;
        for (const k of Object.keys(n)) {
            const v = n[k];
            if (Array.isArray(v)) { if (v.some((it) => !sideEffectFree(it))) return false; }
            else if (v && typeof v === "object") { if (!sideEffectFree(v)) return false; }
        }
        return true;
    }
    for (const n of prog.body) if (n.type === "Func") removeDeadStores(n.body);
    return prog as InstryxNode;
}

/* -------------------------
   Optimization registry & pipeline
   ------------------------- */

type OptPass = { id: string; label: string; run: (ast: InstryxNode, ctx?: any) => InstryxNode };

const defaultPasses: OptPass[] = [
    { id: "ssa", label: "SSA Conversion", run: ssaConvert },
    { id: "constfold", label: "Constant Folding", run: constantFolding },
    { id: "ipcprop", label: "Interprocedural Constant Propagation", run: interproceduralConstantPropagation },
    { id: "peephole", label: "Peephole", run: peepholeOptimize },
    { id: "cfg", label: "CFG Simplify", run: cfgSimplify },
    { id: "inline", label: "Inline Small Functions", run: inlineSmallFunctions },
    { id: "specialize", label: "Function Specialization", run: (ast) => functionSpecialization(ast) },
    { id: "unroll", label: "Loop Unroll", run: loopUnroll },
    { id: "dce", label: "Dead Code Elimination", run: deadCodeElimination },
];

const passRegistry: Map<string, OptPass> = new Map(defaultPasses.map(p => [p.id, p]));

export function registerOptimizationPass(pass: OptPass) { passRegistry.set(pass.id, pass); }

export function listOptimizationPasses(): OptPass[] { return Array.from(passRegistry.values()); }

/**
 * Run optimization pipeline; returns {ast, report}
 */
export function optimizeAst(ast: InstryxNode, opts: RenderOptions = {}): { ast: InstryxNode; report: OptimizationReport } {
    const requested = (opts.optimizePasses && opts.optimizePasses.length > 0) ? opts.optimizePasses : Array.from(passRegistry.keys());
    const report: OptimizationReport = { ranPasses: [], timings: {}, changes: [], astHashBefore: sha1(canonicalJson(ast)), astHashAfter: "" };
    let cur = deepClone(ast);
    for (const pid of requested) {
        const pass = passRegistry.get(pid);
        if (!pass) continue;
        const t0 = Date.now();
        try {
            const before = canonicalJson(cur);
            cur = pass.run(cur, { profile: opts.profile });
            const after = canonicalJson(cur);
            if (before !== after) report.changes.push(`${pass.label}(${pass.id}) applied changes`);
            report.ranPasses.push(pass.id);
        } catch (err) {
            report.changes.push(`PASS ${pass.id} failed: ${(err as Error).message}`);
        } finally {
            report.timings[pass.id] = Date.now() - t0;
        }
    }
    report.astHashAfter = sha1(canonicalJson(cur));
    return { ast: cur, report };
}

/* -------------------------
   Tooling helpers
   ------------------------- */

/**
 * Validate AST surface shape (lightweight)
 */
export function verifyAst(ast: InstryxNode): { ok: boolean; issues: string[] } {
    const issues: string[] = [];
    function visit(n: any, pathStr = "") {
        if (!n || typeof n !== "object") { issues.push(`Node at ${pathStr} is not an object`); return; }
        if (!n.type) issues.push(`Node at ${pathStr} missing type`);
        // some basic checks
        if (n.type === "Func") {
            if (!n.name) issues.push(`Func at ${pathStr} missing name`);
            if (!n.body) issues.push(`Func ${n.name} has no body`);
        }
        for (const k of Object.keys(n)) {
            if (k === "loc" || k === "id" || k === "leadingComments" || k === "trailingComments") continue;
            const v = n[k];
            if (Array.isArray(v)) v.forEach((it, i) => visit(it, `${pathStr}.${k}[${i}]`));
            else if (v && typeof v === "object") visit(v, `${pathStr}.${k}`);
        }
    }
    visit(ast, "root");
    return { ok: issues.length === 0, issues };
}

/**
 * Simple linter that returns warnings/suggestions (non-invasive)
 */
export function lintAst(ast: InstryxNode): { warnings: string[] } {
    const warnings: string[] = [];
    function visit(n: any) {
        if (!n || typeof n !== "object") return;
        if (n.type === "Func") {
            const name = n.name;
            if (!/^[a-zA-Z_]\w*$/.test(name)) warnings.push(`Function name "${name}" is non-idiomatic`);
            const params = n.params ?? [];
            if (params.length > 10) warnings.push(`Function "${name}" has many params (${params.length}), consider refactor`);
        }
        if (n.type === "Binary" && n.op === "==") warnings.push("Use === semantics if available (consider strict equality)");
        for (const k of Object.keys(n)) {
            const v = n[k];
            if (Array.isArray(v)) v.forEach(visit);
            else if (v && typeof v === "object") visit(v);
        }
    }
    visit(ast);
    return { warnings };
}

/**
 * Render AST -> code with optional optimize pipeline and return report
 */
export function renderAst(ast: InstryxNode | InstryxNode[], opts: RenderOptions = {}): RenderResult {
    const state = new RendererState(opts);
    try {
        let program: ProgramNode;
        if (Array.isArray(ast)) program = { type: "Program", body: ast as BaseNode[] };
        else if ((ast as any).type === "Program") program = ast as ProgramNode;
        else program = { type: "Program", body: [ast as BaseNode] };

        let report: OptimizationReport | undefined;
        if (opts.optimize) {
            const res = optimizeAst(program, opts);
            program = res.ast as ProgramNode;
            report = res.report;
        }

        renderNode(program as BaseNode, state);
        const code = state.code;
        const out: RenderResult = { code, optimized: !!opts.optimize };
        if (state.opts.emitSourceMap) out.sourceMap = state.sourceMap;
        if (report) out.report = report;
        return out;
    } catch (err) {
        const fallback = `/* RENDER FAILED: ${(err as Error).message || String(err)} */\n` + JSON.stringify(ast, null, 2);
        return { code: fallback };
    }
}

/* -------------------------
   File helpers & CLI tooling
   ------------------------- */

export async function renderAndOptimizeToFile(inPath: string, outPath?: string, options: RenderOptions & { optimize?: boolean } = {}) {
    const src = await fs.promises.readFile(inPath, "utf8");
    let ast: any;
    try { ast = JSON.parse(src); } catch { throw new Error("Input must be JSON AST"); }
    const result = renderAst(ast, { ...options, optimize: !!options.optimize });
    const dest = outPath ?? inPath.replace(/\.json$/, ".ix");
    await fs.promises.writeFile(dest, result.code, "utf8");
    if (options.emitSourceMap && result.sourceMap) {
        await fs.promises.writeFile(dest + ".map.json", JSON.stringify(result.sourceMap, null, 2), "utf8");
    }
    if (result.report) {
        await fs.promises.writeFile(dest + ".opt-report.json", JSON.stringify(result.report, null, 2), "utf8");
    }
    return { dest, report: result.report };
}

export function watchAndAutoOptimize(inPath: string, outPath?: string, options: RenderOptions & { optimize?: boolean } = {}) {
    let timer: NodeJS.Timeout | null = null;
    const run = async () => {
        try {
            const res = await renderAndOptimizeToFile(inPath, outPath, options);
            console.log(`Rendered ${inPath} -> ${res.dest}`);
            if (res.report) console.log("Optimization report:", res.report);
        } catch (err) {
            console.error("watch error:", (err as Error).message ?? err);
        }
    };
    fs.watch(inPath, () => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => run().catch(console.error), 150);
    });
    run().catch(console.error);
}

/* CLI helper */
export async function renderFileCli(argv: string[]) {
    if (!argv || argv.length < 1) {
        console.error("Usage: renderFileCli <ast.json> [out.ix] [--optimize] [--compact] [--sourcemap]");
        return 2;
    }
    const inPath = argv[0];
    const outPath = argv[1];
    const options: any = {
        optimize: argv.includes("--optimize"),
        compact: argv.includes("--compact"),
        emitSourceMap: argv.includes("--sourcemap"),
        enableIncremental: true,
    };
    try {
        const { dest } = await renderAndOptimizeToFile(inPath, outPath, options);
        console.log("Wrote:", dest);
        return 0;
    } catch (err) {
        console.error("renderFileCli error:", (err as Error).message ?? err);
        return 1;
    }
}

/* -------------------------
   Exports & utilities
   ------------------------- */

export function prettyPrint(ast: InstryxNode | InstryxNode[]) { return renderAst(ast, { indent: "  ", compact: false }).code; }
export function minify(ast: InstryxNode | InstryxNode[]) { return renderAst(ast, { compact: true }).code; }

export function registerNodeHandler(type: string, handler: NodeHandler) { handlers.set(type, handler); }
export function listNodeHandlers(): string[] { return Array.from(handlers.keys()); }
export function listOptimizationPasses(): string[] { return Array.from(passRegistry.keys()); }

export {
    RendererState,
    renderNode,
    handlers as rendererHandlers,
    /* expose optimizations for testing */
    constantFolding,
    ssaConvert,
    cfgSimplify,
    interproceduralConstantPropagation,
    functionSpecialization,
    loopUnroll,
    peepholeOptimize,
    deadCodeElimination,
};

/* -------------------------
   End of file
   ------------------------- */


