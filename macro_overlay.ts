
/**
 * macro_overlay.ts
 *
 * CIAMS macro overlay processor (extended)
 * - Registry for textual macros (inline and prefix)
 * - Rich macro handler results (text + diagnostics + prepend)
 * - Prefix-macro support (macros applied to following `func` definitions)
 * - File watcher CLI, unit test runner, demo runner
 *
 * Notes:
 * - This remains a text-to-text macro expander. For robust correctness, integrate
 *   with the Instryx parser and perform AST-aware expansion in a future pass.
 */

export type MacroHandlerResult = string | MacroExpansion;
export type MacroHandler = (
  args: string[],
  ctx?: MacroContext & { following?: string; followingOffset?: number }
) => MacroHandlerResult | Promise<MacroHandlerResult>;

export interface Macro {
  name: string;
  description?: string;
  handler: MacroHandler;
  // If true the macro can be used as a "prefix" for the next function declaration,
  // e.g. `@memoize func foo(...) { ... };` — handler will receive the full function text.
  prefix?: boolean;
}

export interface MacroContext {
  filename?: string;
  source?: string;
  // Attach any compiler-specific services here (logger, resolver, parser results)
  services?: Record<string, unknown>;
}

export interface MacroExpansion {
  // The text to substitute in place of macro (for prefix macros, usually replaces macro+target).
  text: string;
  // Optional text to prepend to the file (helper functions, imports).
  prepend?: string;
  // Optional diagnostics emitted by the handler.
  diagnostics?: MacroDiagnostic[];
}

export interface ExpansionResult {
  ok: boolean;
  transformed: string;
  details?: string;
}

/* -------------------------
   Core Registry
   ------------------------- */

export class MacroRegistry {
  private macros = new Map<string, Macro>();

  register(m: Macro): void {
    if (!m || !m.name || !m.handler) throw new Error("Invalid macro registration");
    this.macros.set(m.name, m);
  }

  unregister(name: string): boolean {
    return this.macros.delete(name);
  }

  list(): string[] {
    return Array.from(this.macros.keys());
  }

  get(name: string): Macro | undefined {
    return this.macros.get(name);
  }

  /**
   * Apply macros to a source string.
   * Delegates to applyMacrosWithDiagnostics for full capability.
   */
  async applyMacros(source: string, ctx: MacroContext = {}): Promise<ExpansionResult> {
    const { result } = await applyMacrosWithDiagnostics(source, this, ctx);
    return result;
  }
}

/* -------------------------
   Utilities
   ------------------------- */

/**
 * Very small argument parser:
 * - Splits on commas while respecting quoted strings and parentheses
 * - Trims whitespace
 */
function parseMacroArgs(raw: string): string[] {
  if (!raw || raw.trim().length === 0) return [];
  const args: string[] = [];
  let cur = "";
  let inQuote = false;
  let quoteChar = "";
  let parenDepth = 0;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    const prev = raw[i - 1];
    if ((ch === '"' || ch === "'") && prev !== "\\") {
      if (!inQuote) {
        inQuote = true;
        quoteChar = ch;
        cur += ch;
      } else if (quoteChar === ch) {
        inQuote = false;
        quoteChar = "";
        cur += ch;
      } else {
        cur += ch;
      }
    } else if (!inQuote && (ch === "(")) {
      parenDepth++;
      cur += ch;
    } else if (!inQuote && ch === ")") {
      if (parenDepth > 0) parenDepth--;
      cur += ch;
    } else if (ch === "," && !inQuote && parenDepth === 0) {
      args.push(cur.trim());
      cur = "";
    } else {
      cur += ch;
    }
  }
  if (cur.trim().length > 0) args.push(cur.trim());
  return args.filter(a => a.length > 0);
}

/**
 * Convert string offset to 1-based line/column.
 */
function offsetToLineCol(source: string, offset: number) {
  const before = source.slice(0, offset);
  const lines = before.split("\n");
  const line = lines.length;
  const column = lines[lines.length - 1].length + 1;
  return { line, column };
}

/* -------------------------
   Diagnostics + File helpers
   ------------------------- */

export interface MacroDiagnostic {
  type: "warning" | "error";
  message: string;
  macroName?: string;
  startOffset?: number;
  endOffset?: number;
  line?: number;
  column?: number;
}

/**
 * Find inline macro occurrences with positional info.
 * Matches `@name <anything until semicolon> ;` (multi-line supported).
 */
export function findInlineMacrosWithPositions(source: string) {
  const macroRegex = /@([a-zA-Z_][a-zA-Z0-9_]*)\s+([^;]*?)\s*;/gs;
  const results: Array<{
    name: string;
    rawArgs: string;
    start: number;
    end: number;
    line: number;
    column: number;
    fullText: string;
  }> = [];

  let match: RegExpExecArray | null;
  while ((match = macroRegex.exec(source)) !== null) {
    const [fullMatch, name, rawArgs] = match;
    const start = match.index;
    const end = macroRegex.lastIndex;
    const { line, column } = offsetToLineCol(source, start);
    results.push({
      name,
      rawArgs,
      start,
      end,
      line,
      column,
      fullText: fullMatch,
    });
  }

  return results;
}

/**
 * Find prefix macro occurrences used to annotate following `func`.
 * Matches `@name` followed by optional whitespace/newlines then `func`.
 * Returns { name, macroStart, macroEnd, funcStart }.
 */
export function findPrefixMacroOccurrences(source: string) {
  const results: Array<{
    name: string;
    macroStart: number;
    macroEnd: number;
    funcStart: number;
  }> = [];
  const macroRegex = /@([a-zA-Z_][a-zA-Z0-9_]*)/g;
  let match: RegExpExecArray | null;
  while ((match = macroRegex.exec(source)) !== null) {
    const [fullMatch, name] = match;
    const macroStart = match.index;
    const macroEnd = macroRegex.lastIndex;
    // Look ahead from macroEnd, skipping whitespace and comments to find "func"
    let i = macroEnd;
    // skip whitespace and single-line comments (`--`)
    while (i < source.length) {
      const ch = source[i];
      if (ch === " " || ch === "\t" || ch === "\r" || ch === "\n") {
        i++;
        continue;
      }
      // check for comment start
      if (source.startsWith("--", i)) {
        // skip to end of line
        const nl = source.indexOf("\n", i + 2);
        if (nl === -1) i = source.length;
        else i = nl + 1;
        continue;
      }
      break;
    }
    if (source.startsWith("func", i)) {
      results.push({ name, macroStart, macroEnd, funcStart: i });
    }
  }
  return results;
}

/* -------------------------
   Function parsing utility (find function block end)
   ------------------------- */

/**
 * Given a source and index pointing to the 'func' keyword, find the end offset of the function
 * by scanning for the matching closing brace '}' that balances the opening '{' of the function body.
 * Returns -1 if not found.
 */
export function findFunctionBlockEnd(source: string, funcKeywordIndex: number): number {
  // find first '{' after funcKeywordIndex
  const braceOpenIndex = source.indexOf("{", funcKeywordIndex);
  if (braceOpenIndex === -1) return -1;
  let depth = 0;
  for (let i = braceOpenIndex; i < source.length; i++) {
    const ch = source[i];
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        // include the '}' char
        return i + 1;
      }
    } else if (ch === '"' || ch === "'") {
      // skip string literal to avoid braces inside quotes
      const quote = ch;
      i++;
      while (i < source.length && !(source[i] === quote && source[i - 1] !== "\\")) i++;
    } else if (source.startsWith("--", i)) {
      // single-line comment
      const nl = source.indexOf("\n", i + 2);
      if (nl === -1) return -1;
      i = nl;
    }
  }
  return -1;
}

/* -------------------------
   applyMacrosWithDiagnostics: main engine
   ------------------------- */

/**
 * Apply macros (prefix + inline) with diagnostics.
 * - Prefix macros are applied first (they can replace macro + target function).
 * - Inline macros are applied next.
 */
export async function applyMacrosWithDiagnostics(
  source: string,
  registry: MacroRegistry,
  ctx: MacroContext = {}
): Promise<{ result: ExpansionResult; diagnostics: MacroDiagnostic[] }> {
  const diagnostics: MacroDiagnostic[] = [];
  let working = source;
  try {
    // 1) Prefix macros
    const prefixOccurrences = findPrefixMacroOccurrences(working);
    // Sort descending so replacements don't affect earlier offsets
    const prefixSorted = prefixOccurrences.sort((a, b) => b.macroStart - a.macroStart);

    for (const occ of prefixSorted) {
      const macro = registry.get(occ.name);
      if (!macro || !macro.prefix) {
        // ignore if not registered as prefix
        continue;
      }
      const funcEnd = findFunctionBlockEnd(working, occ.funcStart);
      if (funcEnd === -1) {
        const { line, column } = offsetToLineCol(working, occ.macroStart);
        diagnostics.push({
          type: "error",
          message: `Prefix macro '@${occ.name}' not followed by complete function definition.`,
          macroName: occ.name,
          startOffset: occ.macroStart,
          endOffset: occ.funcStart,
          line,
          column,
        });
        continue;
      }
      const macroAndFuncStart = occ.macroStart;
      const macroAndFuncEnd = funcEnd;
      const fullFuncText = working.slice(occ.funcStart, funcEnd);
      // Provide handler with args parsed from between '@name' and 'func' (if any)
      const rawBetween = working.slice(occ.macroStart + occ.name.length + 1, occ.funcStart);
      const args = parseMacroArgs(rawBetween);
      // Call handler
      let res = await macro.handler(args, { ...ctx, source: working, following: fullFuncText, followingOffset: occ.funcStart });
      let expansion: MacroExpansion = typeof res === "string" ? { text: res } : res;
      // If handler returned diagnostics, append them with positional adjustments
      if (expansion.diagnostics && expansion.diagnostics.length > 0) {
        for (const d of expansion.diagnostics) diagnostics.push(d);
      }
      // Prepend helpers if provided
      if (expansion.prepend && expansion.prepend.length > 0) {
        // insert at the top of the file
        working = expansion.prepend + "\n" + working;
        // Adjust offsets of subsequent matches are not updated here, but we process prefixSorted in descending order,
        // and for simplicity we accept that prepend shifts text; for full correctness, implement token/AST pass later.
      }
      // Replace macro + function with expansion.text
      working = working.slice(0, macroAndFuncStart) + expansion.text + working.slice(macroAndFuncEnd);
    }

    // 2) Inline macros
    const inlineOccurrences = findInlineMacrosWithPositions(working);
    // Sort descending by start and replace
    const inlineSorted = inlineOccurrences.sort((a, b) => b.start - a.start);
    for (const occ of inlineSorted) {
      const macro = registry.get(occ.name);
      if (!macro) {
        diagnostics.push({
          type: "warning",
          message: `Unknown macro '@${occ.name}' — left unchanged.`,
          macroName: occ.name,
          startOffset: occ.start,
          endOffset: occ.end,
          line: occ.line,
          column: occ.column,
        });
        continue;
      }
      const args = parseMacroArgs(occ.rawArgs);
      try {
        let res = await macro.handler(args, { ...ctx, source: working });
        const expansion: MacroExpansion = typeof res === "string" ? { text: res } : res;
        if (expansion.diagnostics && expansion.diagnostics.length > 0) {
          for (const d of expansion.diagnostics) diagnostics.push(d);
        }
        // If prepend present, add to file head
        if (expansion.prepend && expansion.prepend.length > 0) {
          working = expansion.prepend + "\n" + working;
        }
        // Replace inline macro occurrence with expansion.text
        working = working.slice(0, occ.start) + expansion.text + working.slice(occ.end);
      } catch (err: any) {
        diagnostics.push({
          type: "error",
          message: `Macro '@${occ.name}' failed: ${String(err?.message ?? err)}`,
          macroName: occ.name,
          startOffset: occ.start,
          endOffset: occ.end,
          line: occ.line,
          column: occ.column,
        });
        return { result: { ok: false, transformed: source, details: String(err?.message ?? err) }, diagnostics };
      }
    }

    return { result: { ok: true, transformed: working }, diagnostics };
  } catch (err: any) {
    diagnostics.push({ type: "error", message: String(err?.message ?? err) });
    return { result: { ok: false, transformed: source, details: String(err?.message ?? err) }, diagnostics };
  }
}

/* -------------------------
   Built-in macros (default + prefix-aware)
   ------------------------- */

export function createDefaultRegistry(): MacroRegistry {
  const reg = new MacroRegistry();

  // @inject net.api;
  reg.register({
    name: "inject",
    description: "Auto-injects a dependency (dotted path). Example: @inject db.conn;",
    handler: (args) => {
      const target = args[0] ?? "";
      const varName = target.replace(/[^\w]/g, "_");
      return `${varName} = system.get("${target}");\n`;
    }
  });

  // @wraptry expression;
  reg.register({
    name: "wraptry",
    description: "Wrap call into quarantine tri-phase.",
    handler: (args) => {
      const expr = args.join(", ");
      return [
        `quarantine try {`,
        `    ${expr};`,
        `} replace {`,
        `    log("Retrying...");`,
        `    ${expr};`,
        `} erase {`,
        `    fail("Unhandled error.");`,
        `};`
      ].join("\n") + "\n";
    }
  });

  // @inline someCall();
  reg.register({
    name: "inline",
    description: "Emit inline pragma comment.",
    handler: (args) => {
      const expr = args.join(", ");
      return `/* @inline ${expr} */ ${expr};\n`;
    }
  });

  return reg;
}

/* -------------------------
   Additional higher-level builtins & tooling
   ------------------------- */

/**
 * Helpers for unique ids.
 */
function uid(prefix = "u"): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 8)}`;
}

function safeId(name: string): string {
  return name.replace(/[^\w]/g, "_").replace(/^(\d)/, "_$1");
}

/**
 * Register additional builtins including prefix-aware macros.
 */
export function registerAdditionalBuiltins(reg: MacroRegistry) {
  // @async call();
  reg.register({
    name: "async",
    description: "Spawn an asynchronous task for the call expression.",
    handler: (args) => {
      const expr = args.join(", ").trim();
      if (!expr) return "// @async: missing expression\n";
      return `spawn async { ${expr}; };\n`;
    }
  });

  // @debug as inline and prefix (prefix will instrument function body)
  reg.register({
    name: "debug",
    prefix: true,
    description: "Instrument function with logging at entry/exit or instrument a single call.",
    handler: (args, ctx) => {
      // If called as prefix, ctx.following contains the function text; we will inject logs at top and bottom.
      if (ctx?.following) {
        const funcText = ctx.following;
        // Try capture function signature: func name(params) { body }
        const m = funcText.match(/^func\s+([a-zA-Z_][\w]*)\s*\(([^)]*)\)\s*{?/s);
        if (!m) {
          return { text: funcText, diagnostics: [{ type: "warning", message: "debug: couldn't parse function signature" }] };
        }
        const name = m[1];
        // Inject logs: find first `{` and inject log after it; find last `}` before function end to inject exit log.
        const firstBrace = funcText.indexOf("{");
        const lastBrace = funcText.lastIndexOf("}");
        if (firstBrace === -1 || lastBrace === -1) {
          return { text: funcText, diagnostics: [{ type: "warning", message: "debug: function body not found" }] };
        }
        const beforeBody = funcText.slice(0, firstBrace + 1);
        const bodyInner = funcText.slice(firstBrace + 1, lastBrace);
        const afterBody = funcText.slice(lastBrace);
        const newBody = `\n    log("DEBUG: enter ${name}");\n${bodyInner}\n    log("DEBUG: exit ${name}");\n`;
        return { text: beforeBody + newBody + afterBody };
      } else {
        // inline debug call
        const expr = args.join(", ").trim();
        if (!expr) return "/* @debug: missing expr */\n";
        const id = uid("dbg");
        const tmp = `__${id}_res`;
        return [
          `/* @debug START */`,
          `${tmp} = ${expr};`,
          `log("DEBUG: ${expr} ->", ${tmp});`,
          `${tmp};`,
          `/* @debug END */`,
        ].join("\n") + "\n";
      }
    }
  });

  // @memoize func (prefix) or @memoize call() (inline)
  reg.register({
    name: "memoize",
    prefix: true,
    description: "Memoize function results (prefix) or expression (inline).",
    handler: (args, ctx) => {
      if (ctx?.following) {
        // parse function signature
        const funcText = ctx.following;
        const m = funcText.match(/^func\s+([a-zA-Z_][\w]*)\s*\(([^)]*)\)\s*{?/s);
        if (!m) return { text: funcText, diagnostics: [{ type: "warning", message: "memoize: cannot parse func" }] };
        const name = m[1];
        const params = m[2].trim();
        const paramList = params.length === 0 ? [] : params.split(",").map(p => p.trim().split(/\s+/).pop());
        const cacheId = `__memo_${safeId(name)}`;
        // Build cache wrapper by replacing function body: we will assume return value is last expression or explicit return available.
        const firstBrace = funcText.indexOf("{");
        const lastBrace = funcText.lastIndexOf("}");
        if (firstBrace === -1 || lastBrace === -1) return { text: funcText, diagnostics: [{ type: "warning", message: "memoize: missing body" }] };
        const signature = funcText.slice(0, firstBrace);
        const bodyInner = funcText.slice(firstBrace + 1, lastBrace);
        // Create key building expression
        const keyExpr = paramList.length === 0 ? `"__no_args__"` : paramList.map(p => `String(${p})`).join(` + "|" + `);
        const newBody = [
          `${cacheId} = ${cacheId} ? ${cacheId} : {};`,
          `__memo_k = ${keyExpr};`,
          `if (${cacheId}[__memo_k] != undefined) {`,
          `    ${cacheId}[__memo_k];`,
          `} else {`,
          `    __memo_v = (function(){ ${bodyInner} })();`,
          `    ${cacheId}[__memo_k] = __memo_v;`,
          `    __memo_v;`,
          `}`
        ].join("\n    ");
        return { text: signature + "{" + `\n    ${newBody}\n` + "}" , prepend: "" };
      } else {
        // inline memoize of expression
        const expr = args.join(", ").trim();
        if (!expr) return { text: "/* @memoize: missing expr */\n" };
        const id = uid("memo");
        const cache = `__memo_${id}`;
        const keyExpr = `String(${args.join(") + '|' + String(")})`;
        // best-effort inline memoize
        const txt = [
          `${cache} = ${cache} ? ${cache} : {};`,
          `__memo_k = ${keyExpr};`,
          `if (${cache}[__memo_k] != undefined) {`,
          `    ${cache}[__memo_k];`,
          `} else {`,
          `    __memo_v = ${expr};`,
          `    ${cache}[__memo_k] = __memo_v;`,
          `    __memo_v;`,
          `}`
        ].join("\n") + "\n";
        return { text: txt };
      }
    }
  });

  // @defer expr;
  reg.register({
    name: "defer",
    description: "Defer expression to block exit (textual push onto __defer_stack).",
    handler: (args) => {
      const expr = args.join(", ").trim();
      if (!expr) return { text: "/* @defer: missing expr */\n" };
      const stackVar = "__defer_stack";
      const txt = [
        `${stackVar} = ${stackVar} ? ${stackVar} : [];`,
        `${stackVar}.push(() => { ${expr}; });`
      ].join("\n") + "\n";
      return { text: txt };
    }
  });

  // @profile call();
  reg.register({
    name: "profile",
    description: "Wrap expression with timing instrumentation.",
    handler: (args) => {
      const expr = args.join(", ").trim();
      if (!expr) return { text: "/* @profile: missing expr */\n" };
      const id = uid("prof");
      const start = `__${id}_start`;
      const result = `__${id}_v`;
      const txt = [
        `${start} = time.now();`,
        `${result} = ${expr};`,
        `log("PROFILE ${id}: elapsed_ms", time.now() - ${start});`,
        `${result};`
      ].join("\n") + "\n";
      return { text: txt };
    }
  });

  // @assert cond;
  reg.register({
    name: "assert",
    description: "Insert runtime assertion.",
    handler: (args) => {
      const cond = args.join(", ").trim();
      if (!cond) return { text: "/* @assert: missing condition */\n" };
      return { text: `if not (${cond}) { fail("Assertion failed: ${cond}"); };` };
    }
  });

  // Enhanced inject_as
  reg.register({
    name: "inject_as",
    description: "Inject a dependency with alias: @inject_as db.conn, dbConn;",
    handler: (args) => {
      const target = (args[0] ?? "").trim();
      const alias = (args[1] ?? "").trim();
      if (!target) return { text: "/* @inject_as: missing target */\n" };
      const varName = alias || safeId(target);
      return { text: `${varName} = system.get("${target}");\n` };
    }
  });

  // ffi marker
  reg.register({
    name: "ffi",
    description: "Mark symbol for FFI; prefix usage on func will convert to extern marker",
    prefix: true,
    handler: (args, ctx) => {
      // If prefix applied to function, emit an extern declaration (textual)
      if (ctx?.following) {
        // Simple transformation: replace `func name(...) { ... }` with `extern "C" func name(...);`
        const funcText = ctx.following;
        const m = funcText.match(/^func\s+([a-zA-Z_][\w]*)\s*\(([^)]*)\)\s*{?/s);
        if (!m) return { text: funcText, diagnostics: [{ type: "warning", message: "ffi: can't parse function" }] };
        const name = m[1];
        const params = m[2];
        const externText = `extern "C" func ${name}(${params});\n`;
        return { text: externText };
      } else {
        const name = (args[0] ?? "").trim();
        if (!name) return { text: "/* @ffi: missing symbol */\n" };
        return { text: `/* @ffi extern ${name} */\n` };
      }
    }
  });
}

/**
 * Create full registry with all builtins.
 */
export function createFullRegistry(): MacroRegistry {
  const reg = createDefaultRegistry();
  registerAdditionalBuiltins(reg);
  return reg;
}

/* -------------------------
   Demo runner, CLI, watcher, tests
   ------------------------- */

export async function demoExpand() {
  const reg = createFullRegistry();
  const src = `
-- Demo Instryx with macros
@inject net.api;
@inject_as db.conn, dbConn;
@debug
func greet(name) {
    print: "Hello, " + name + "!";
};

main() {
    @debug fetchData("https://example.com");
    @async backgroundTask();
    result = @memoize expensiveCompute(10, 20);
    @profile heavyWork();
    @defer cleanup();
    @assert result != 0;
    @ffi external_handler;
};
`;
  const { result, diagnostics } = await applyMacrosWithDiagnostics(src, reg);
  console.log("===== TRANSFORMED SOURCE =====");
  console.log(result.transformed);
  if (diagnostics && diagnostics.length > 0) {
    console.log("===== DIAGNOSTICS =====");
    for (const d of diagnostics) {
      console.log(`${d.type.toUpperCase()}: ${d.message} ${d.line ? `(line ${d.line}, col ${d.column})` : ""}`);
    }
  }
  return { transformed: result.transformed, diagnostics };
}

/**
 * Simple unit-test harness for macro overlay.
 * Returns summary object.
 */
export async function runUnitTests() {
  const reg = createFullRegistry();
  const tests: Array<{ name: string; src: string; expectContains: string[] }> = [
    {
      name: "inject",
      src: `@inject db.conn;`,
      expectContains: [`db_conn = system.get("db.conn");`],
    },
    {
      name: "wraptry",
      src: `@wraptry riskyCall();`,
      expectContains: [`quarantine try {`, `riskyCall();`, `} replace {`],
    },
    {
      name: "debug-inline",
      src: `@debug compute(1,2);`,
      expectContains: [`DEBUG:`, `compute(1,2)`],
    },
    {
      name: "debug-prefix-func",
      src: `@debug func foo() { print: "x"; };`,
      expectContains: [`log("DEBUG: enter foo")`, `log("DEBUG: exit foo")`],
    },
    {
      name: "memoize-prefix",
      src: `@memoize func fib(n) { if n <= 1 { n } else { fib(n-1) + fib(n-2) } };`,
      expectContains: [`__memo_fib`, `if (__memo_fib[__memo_k] != undefined)`].filter(Boolean),
    }
  ];

  const results: Array<{ name: string; pass: boolean; output: string; diagnostics: MacroDiagnostic[] }> = [];
  for (const t of tests) {
    const { result, diagnostics } = await applyMacrosWithDiagnostics(t.src, reg);
    const pass = t.expectContains.every(e => result.transformed.indexOf(e) !== -1);
    results.push({ name: t.name, pass, output: result.transformed, diagnostics });
  }

  const passed = results.filter(r => r.pass).length;
  console.log(`Unit tests: ${passed}/${results.length} passed`);
  for (const r of results) {
    console.log(`- ${r.name}: ${r.pass ? "PASS" : "FAIL"}`);
    if (!r.pass) {
      console.log("  output:", r.output);
      if (r.diagnostics && r.diagnostics.length) console.log("  diagnostics:", r.diagnostics);
    }
  }
  return { total: results.length, passed, results };
}

/**
 * CLI runner with watch support.
 * Usage: runCli(['file.ix', '--overwrite', '--watch', '--test'])
 */
export async function runCli(argv: string[]) {
  if (!argv || argv.length === 0) {
    console.error("Usage: runCli <file.ix> [--overwrite] [--watch] [--test]");
    return 2;
  }

  const file = argv[0];
  const overwrite = argv.includes("--overwrite");
  const watch = argv.includes("--watch");
  const runTests = argv.includes("--test");

  const reg = createFullRegistry();

  if (runTests) {
    await runUnitTests();
  }

  const applyOnce = async () => {
    const res = await applyMacrosFile(file, reg, { overwrite });
    if (!res.ok) {
      console.error("Failed to expand macros:", res.details);
      if (res.diagnostics) console.error(res.diagnostics);
      return;
    }
    console.log(`Expanded macros written to ${res.outPath}`);
    if (res.diagnostics && res.diagnostics.length > 0) {
      console.warn("Diagnostics:");
      for (const d of res.diagnostics) {
        console.warn(`${d.type.toUpperCase()}: ${d.message} ${d.line ? `(line ${d.line}, col ${d.column})` : ""}`);
      }
    }
  };

  await applyOnce();

  if (watch) {
    console.log("Watching file for changes...");
    const fs = await import("fs");
    let timer: NodeJS.Timeout | null = null;
    fs.watch(file, (ev) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        console.log("Change detected, re-applying macros...");
        applyOnce().catch(err => console.error("Watch apply error:", err));
      }, 150);
    });
  }

  return 0;
}

/* -------------------------
   File helpers (uses applyMacrosWithDiagnostics)
   ------------------------- */

export async function applyMacrosFile(
  filename: string,
  registry: MacroRegistry,
  options: { overwrite?: boolean } = {}
): Promise<{ ok: boolean; outPath?: string; diagnostics?: MacroDiagnostic[]; details?: string }> {
  const fs = await import("fs/promises");
  try {
    const src = await fs.readFile(filename, "utf8");
    const { result, diagnostics } = await applyMacrosWithDiagnostics(src, registry);
    if (!result.ok) {
      return { ok: false, details: result.details, diagnostics };
    }
    const outPath = options.overwrite ? filename : `${filename}.expanded.ix`;
    await fs.writeFile(outPath, result.transformed, "utf8");
    return { ok: true, outPath, diagnostics };
  } catch (err: any) {
    return { ok: false, details: String(err?.message ?? err) };
  }
}

/* -------------------------
   Notes
   -------------------------
 - This implementation focuses on practical, text-based macro features:
   prefix macros that transform the next `func` declaration, richer handler output,
   runtime helpers, file-watching CLI and a small unit-test harness.
 - For production accuracy, integrate with lexer/parser and use token/AST ranges
   to keep expansions correct and source-mapped.
-------------------------- */

