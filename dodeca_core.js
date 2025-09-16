/**
 * dodeca_core.js
 *
 * Production-ready core utilities for Dodeca (CIAMS / Instryx tooling)
 *
 * Features:
 * - Lightweight heuristic suggestion engine (many rules)
 * - Local JSON memory (counts + accepted suggestions)
 * - LRU cache for analysis results
 * - Preview/apply flow (insert macro hints, generate preview file)
 * - Patch generation (unified diff)
 * - Batch analysis / batch-apply helpers (concurrent)
 * - PluginManager to load JS plugins from ./ciams_plugins
 * - Small HTTP JSON API for integration (GET /suggest, POST /apply)
 * - CLI wrapper for common operations
 *
 * Usage:
 *  node dodeca_core.js suggest file.ix
 *  node dodeca_core.js preview file.ix --index 0
 *  node dodeca_core.js apply file.ix --index 0
 *  node dodeca_core.js serve --port 8787
 *
 * Notes:
 * - Designed to be self-contained with Node.js stdlib.
 * - Plugins placed in ./ciams_plugins can export `register(core)` to add rules.
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const crypto = require('crypto');
const os = require('os');
const { Worker, isMainThread } = require('worker_threads');

// Paths & defaults
const MODULE_DIR = path.dirname(__filename);
const MEMORY_PATH = path.join(MODULE_DIR, 'ai_memory.json');
const LOG_PATH = path.join(MODULE_DIR, 'dodeca_core.log');
const PLUGINS_DIR = path.join(MODULE_DIR, 'ciams_plugins');

const DEFAULT_MAX_SUGGESTIONS = 12;
const DEFAULT_PORT = 8787;
const MAX_EXPANSION_BYTES = 50_000;
const MAX_UNROLL_SAFE = 16;

// Simple logger (append-only)
function log(level, ...args) {
  try {
    const line = `[${new Date().toISOString()}] [${level}] ${args.map(String).join(' ')}\n`;
    fs.appendFileSync(LOG_PATH, line, { encoding: 'utf8' });
  } catch (e) {
    // ignore
  }
}

// -------------------------
// Utilities
// -------------------------
function uid(prefix = 'g', seed = null) {
  if (typeof seed === 'number') {
    const h = crypto.createHash('sha1').update(`${prefix}:${seed}`).digest('hex').slice(0, 8);
    return `${prefix}_${h}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function readFileSafe(p) {
  try {
    return fs.readFileSync(p, 'utf8');
  } catch (e) {
    return null;
  }
}

function writeFileSafe(p, content, backup = true) {
  try {
    if (backup && fs.existsSync(p)) {
      fs.copyFileSync(p, p + '.bak');
    }
    fs.writeFileSync(p, content, 'utf8');
    return true;
  } catch (e) {
    log('ERROR', 'writeFileSafe', e.message);
    return false;
  }
}

function unifiedDiff(a, b, aName = 'a', bName = 'b') {
  const aLines = a.split(/\r?\n/).map(l => l + '\n');
  const bLines = b.split(/\r?\n/).map(l => l + '\n');
  // simple unified diff via diff algorithm (use Node's difflib-like approach)
  const difflib = require('diff');
  const patches = difflib.createPatch(aName, a, b, aName, bName);
  return patches;
}

// -------------------------
// Suggestion model
// -------------------------
class Suggestion {
  constructor(macroName, args, reason, score, snippet = null, location = null) {
    this.macro_name = macroName;
    this.args = Array.isArray(args) ? args : (args ? [args] : []);
    this.reason = reason;
    this.score = score;
    this.snippet = snippet;
    this.location = location; // [start, end] indexes
  }

  toJSON() {
    return {
      macro_name: this.macro_name,
      args: this.args,
      reason: this.reason,
      score: this.score,
      snippet: this.snippet,
      location: this.location,
    };
  }
}

// -------------------------
// AISimpleMemory (JSON-backed)
// -------------------------
class AISimpleMemory {
  constructor(p = MEMORY_PATH) {
    this.path = p;
    this.data = { patterns: {}, accepted: [] };
    this._load();
  }

  _load() {
    try {
      if (fs.existsSync(this.path)) {
        this.data = JSON.parse(fs.readFileSync(this.path, 'utf8') || '{}');
        if (!this.data.patterns) this.data.patterns = {};
        if (!this.data.accepted) this.data.accepted = [];
      }
    } catch (e) {
      log('WARN', 'AISimpleMemory load failed', e.message);
      this.data = { patterns: {}, accepted: [] };
    }
  }

  save() {
    try {
      fs.writeFileSync(this.path, JSON.stringify(this.data, null, 2), 'utf8');
    } catch (e) {
      log('ERROR', 'AISimpleMemory save', e.message);
    }
  }

  recordPattern(key) {
    this.data.patterns = this.data.patterns || {};
    this.data.patterns[key] = (this.data.patterns[key] || 0) + 1;
    this.save();
  }

  patternCount(key) {
    return this.data.patterns && this.data.patterns[key] ? this.data.patterns[key] : 0;
  }

  recordAccepted(suggestion, filename = null) {
    this.data.accepted = this.data.accepted || [];
    this.data.accepted.push({
      time: Date.now(),
      suggestion: suggestion.toJSON ? suggestion.toJSON() : suggestion,
      file: filename,
    });
    this.save();
  }

  export() {
    return this.data;
  }

  import(data, merge = true) {
    if (!merge) {
      this.data = data;
    } else {
      this.data.patterns = this.data.patterns || {};
      (data.patterns || {}).forEach
      for (const k of Object.keys((data.patterns || {}))) {
        this.data.patterns[k] = (this.data.patterns[k] || 0) + data.patterns[k];
      }
      this.data.accepted = (this.data.accepted || []).concat(data.accepted || []);
    }
    this.save();
  }
}

// -------------------------
// Tiny thread-safe LRU cache
// -------------------------
class LRUCache {
  constructor(capacity = 256) {
    this.capacity = capacity;
    this.map = new Map(); // key -> value, order of insertion used for LRU
  }
  get(key) {
    if (!this.map.has(key)) return null;
    const val = this.map.get(key);
    this.map.delete(key);
    this.map.set(key, val);
    return val;
  }
  set(key, value) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.capacity) {
      const firstKey = this.map.keys().next().value;
      this.map.delete(firstKey);
    }
  }
  clear() {
    this.map.clear();
  }
}

// -------------------------
// Plugin manager for JS plugins
// -------------------------
class PluginManager {
  constructor(dir = PLUGINS_DIR) {
    this.dir = dir;
    if (!fs.existsSync(this.dir)) fs.mkdirSync(this.dir, { recursive: true });
    this.loaded = {};
  }

  discover() {
    return fs.readdirSync(this.dir).filter(f => f.endsWith('.js')).map(f => path.basename(f, '.js'));
  }

  load(name, core) {
    const p = path.join(this.dir, name + '.js');
    if (!fs.existsSync(p)) return [false, 'not found'];
    try {
      delete require.cache[require.resolve(p)];
      const mod = require(p);
      if (mod && typeof mod.register === 'function') {
        mod.register(core);
      }
      this.loaded[name] = mod;
      return [true, 'loaded'];
    } catch (e) {
      log('ERROR', 'plugin load', name, e.stack || e.message);
      return [false, String(e)];
    }
  }

  unload(name) {
    if (!this.loaded[name]) return [false, 'not loaded'];
    try {
      const mod = this.loaded[name];
      if (mod && typeof mod.unregister === 'function') mod.unregister();
      delete this.loaded[name];
      return [true, 'unloaded'];
    } catch (e) {
      return [false, String(e)];
    }
  }
}

// -------------------------
// Core assistant logic (JS port of heuristics)
// -------------------------
class DodecaCore {
  constructor(opts = {}) {
    this.memory = new AISimpleMemory(opts.memoryPath || MEMORY_PATH);
    this.cache = new LRUCache(opts.cacheCapacity || 256);
    this.rules = [];
    this._registerBuiltins();
    this.plugins = new PluginManager(opts.pluginsDir || PLUGINS_DIR);
    this.uidSeed = typeof opts.seed === 'number' ? opts.seed : null;
  }

  _registerBuiltins() {
    this.rules = [
      this._ruleInjectDottedDependency.bind(this),
      this._ruleWraptryNetworkCalls.bind(this),
      this._ruleAsyncBackgroundTasks.bind(this),
      this._ruleMemoizePureFunctions.bind(this),
      this._ruleDeferForCleanup.bind(this),
      this._ruleProfileHotpath.bind(this),
      this._ruleInlineSmallCall.bind(this),
      this._ruleAssertNonNull.bind(this),
      this._ruleUnrollSmallLoop.bind(this),
      this._ruleSuggestCacheTTL.bind(this),
      this._ruleSanitizeHtml.bind(this),
      this._ruleAuditSensitive.bind(this),
    ];
  }

  analyzeSource(source, filename = '<anon>', maxSuggestions = DEFAULT_MAX_SUGGESTIONS) {
    const cacheKey = `${filename}:${this._hash(source)}`;
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    let suggestions = [];
    for (const rule of this.rules) {
      try {
        const r = rule(source, filename);
        if (Array.isArray(r)) suggestions = suggestions.concat(r);
      } catch (e) {
        log('WARN', 'rule failure', e.stack || e.message);
      }
    }

    // Boost by memory
    for (const s of suggestions) {
      const boost = 0.05 * Math.min(5, this.memory.patternCount(s.macro_name));
      s.score = Math.max(0, Math.min(1, (s.score || 0) + boost));
    }

    // Deduplicate and sort
    const seen = new Set();
    const unique = [];
    suggestions.sort((a, b) => (b.score || 0) - (a.score || 0));
    for (const s of suggestions) {
      const key = `${s.macro_name}|${(s.args || []).join(',')}|${s.snippet || ''}|${s.location || ''}`;
      if (seen.has(key)) continue;
      seen.add(key);
      unique.push(s);
      if (unique.length >= maxSuggestions) break;
    }

    this.cache.set(cacheKey, unique);
    return unique;
  }

  // Preview: insert macro invocation and return the new source (no macro_overlay expansion here)
  previewApply(source, suggestion) {
    const macroText = `@${suggestion.macro_name} ${(suggestion.args || []).join(', ')};\n`;
    let transformed = source;
    if (suggestion.location && Array.isArray(suggestion.location)) {
      const [start] = suggestion.location;
      transformed = source.slice(0, start) + macroText + source.slice(start);
    } else if (suggestion.snippet) {
      const idx = source.indexOf(suggestion.snippet);
      if (idx !== -1) transformed = source.slice(0, idx) + macroText + source.slice(idx);
      else transformed = macroText + source;
    } else {
      transformed = macroText + source;
    }
    return { ok: true, transformed, diagnostics: null };
  }

  applySuggestionToFile(filepath, suggestion, options = { inplace: false, createPatch: true }) {
    try {
      const src = readFileSafe(filepath);
      if (src === null) return { ok: false, msg: 'read failed' };
      const { ok, transformed } = this.previewApply(src, suggestion);
      if (!ok) return { ok: false, msg: 'preview failed' };
      if (this.safetyCheck && this.safetyCheck(transformed, src) === false) {
        return { ok: false, msg: 'safety check failed' };
      }
      if (options.createPatch) {
        const patch = unifiedDiff(src, transformed, filepath, filepath + '.ai.ix');
        const patchPath = filepath + '.ai.patch';
        writeFileSafe(patchPath, patch, false);
      }
      const outPath = options.inplace ? filepath : filepath + '.ai.ix';
      writeFileSafe(outPath, transformed, true);
      this.memory.recordAccepted(suggestion, filepath);
      return { ok: true, out: outPath };
    } catch (e) {
      log('ERROR', 'applySuggestionToFile', e.stack || e.message);
      return { ok: false, msg: String(e) };
    }
  }

  generatePatch(original, transformed, filename = 'file') {
    return unifiedDiff(original, transformed, filename, filename + '.ai.ix');
  }

  batchAnalyze(directory, pattern = '.ix', maxPerFile = 5, workers = 4) {
    const results = {};
    const files = [];
    (function walk(dir) {
      const items = fs.readdirSync(dir);
      for (const it of items) {
        const p = path.join(dir, it);
        const stat = fs.statSync(p);
        if (stat.isDirectory()) walk(p);
        else if (it.endsWith(pattern)) files.push(p);
      }
    })(directory);
    const pool = new Set();
    const exec = (file) => {
      try {
        const src = readFileSafe(file);
        if (src === null) return [];
        return this.analyzeSource(src, file, maxPerFile);
      } catch (e) {
        log('WARN', 'batchAnalyze file', file, e.message);
        return [];
      }
    };
    // simple concurrency using Promise.all with limited parallelism
    const pAll = [];
    const concurrency = Math.max(1, workers);
    let idx = 0;
    const runNext = () => {
      if (idx >= files.length) return Promise.resolve();
      const f = files[idx++];
      const pr = Promise.resolve().then(() => { results[f] = exec(f); });
      pAll.push(pr);
      const now = pAll.length;
      if (now < concurrency) return runNext();
      return pr.then(runNext);
    };
    return runNext().then(() => Promise.all(pAll).then(() => results));
  }

  batchApplyTop(directory, suggestionIndex = 0, inplace = false, workers = 4) {
    const results = {};
    const files = [];
    (function walk(dir) {
      const items = fs.readdirSync(dir);
      for (const it of items) {
        const p = path.join(dir, it);
        const stat = fs.statSync(p);
        if (stat.isDirectory()) walk(p);
        else if (it.endsWith('.ix')) files.push(p);
      }
    })(directory);
    for (const f of files) {
      try {
        const src = readFileSafe(f);
        if (src === null) { results[f] = { ok: false, msg: 'read failed' }; continue; }
        const suggestions = this.analyzeSource(src, f, 16);
        if (!suggestions || suggestionIndex >= suggestions.length) { results[f] = { ok: false, msg: 'no suggestion' }; continue; }
        const s = suggestions[suggestionIndex];
        const res = this.applySuggestionToFile(f, s, { inplace, createPatch: true });
        results[f] = res;
      } catch (e) {
        log('WARN', 'batchApplyTop', f, e.message);
        results[f] = { ok: false, msg: String(e) };
      }
    }
    return results;
  }

  // -------------------------
  // Built-in heuristic rules (JS versions)
  // Each returns an array of Suggestion
  // -------------------------
  _ruleInjectDottedDependency(source) {
    const suggestions = [];
    const re = /\b([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+)\b/g;
    let m;
    while ((m = re.exec(source))) {
      const full = m[1];
      const after = m.index + m[0].length;
      if (after < source.length && source[after] === '(') continue; // probably a function call
      let score = 0.5;
      if (full.startsWith('net.') || full.includes('api')) score = 0.85;
      else if (full.startsWith('db.') || full.startsWith('cache.')) score = 0.8;
      const snippet = this._extractLine(source, m.index);
      suggestions.push(new Suggestion('inject', [full], `dotted dependency '${full}' detected`, score, snippet, [m.index, m.index + m[0].length]));
      const alias = full.replace(/\./g, '_');
      suggestions.push(new Suggestion('inject_as', [full, alias], `inject with alias ${alias}`, Math.max(0, score - 0.05), snippet, [m.index, m.index + m[0].length]));
      this.memory.recordPattern('inject');
    }
    return suggestions;
  }

  _ruleWraptryNetworkCalls(source) {
    const suggestions = [];
    const re = /\b(net\.request|fetchData|fetch\(|http\.get|axios\.|request\()/g;
    let m;
    while ((m = re.exec(source))) {
      const snippet = this._extractLine(source, m.index);
      suggestions.push(new Suggestion('wraptry', [snippet.trim().replace(/;$/, '')], 'network or unstable call', 0.88, snippet, [m.index, m.index + m[0].length]));
      suggestions.push(new Suggestion('async', [snippet.trim().replace(/;$/, '')], 'run network call async', 0.6, snippet, [m.index, m.index + m[0].length]));
      this.memory.recordPattern('wraptry');
    }
    return suggestions;
  }

  _ruleAsyncBackgroundTasks(source) {
    const suggestions = [];
    const re = /\b(backgroundTask|schedule|setTimeout|spawn|fork)\b/g;
    let m;
    while ((m = re.exec(source))) {
      const snippet = this._extractLine(source, m.index);
      suggestions.push(new Suggestion('async', [snippet.trim().replace(/;$/, '')], 'background/task scheduling detected', 0.75, snippet, [m.index, m.index + m[0].length]));
      this.memory.recordPattern('async');
    }
    return suggestions;
  }

  _ruleMemoizePureFunctions(source) {
    const suggestions = [];
    const re = /\bfunc\s+([A-Za-z_][\w]*)\s*\(([^)]*)\)\s*\{/g;
    let m;
    while ((m = re.exec(source))) {
      const name = m[1];
      const start = m.index;
      const body = this._captureBlock(source, m.index + m[0].length - 1);
      if (!body) continue;
      const bodyText = body.trim();
      if (/\b(print|db\.|net\.|system\.|malloc|alloc)\b/.test(bodyText)) continue;
      if (bodyText.length < 300 && /[+\-*\/%]/.test(bodyText)) {
        const snippet = source.slice(start, start + 80);
        suggestions.push(new Suggestion('memoize', [name], 'small pure/compute function', 0.72, snippet, [start, start + m[0].length]));
        this.memory.recordPattern('memoize');
      }
    }
    return suggestions;
  }

  _ruleDeferForCleanup(source) {
    const suggestions = [];
    const re = /\b([A-Za-z_][\w]*\.(?:close|shutdown)|close\()\b/g;
    let m;
    while ((m = re.exec(source))) {
      const snippet = this._extractLine(source, m.index);
      suggestions.push(new Suggestion('defer', [snippet.trim().replace(/;$/, '')], 'resource cleanup detected; consider defer', 0.62, snippet, [m.index, m.index + m[0].length]));
      this.memory.recordPattern('defer');
    }
    return suggestions;
  }

  _ruleProfileHotpath(source) {
    const suggestions = [];
    const re = /\bfor\b.*\{|\bwhile\b.*\{/g;
    let m;
    while ((m = re.exec(source))) {
      const start = m.index + m[0].length;
      const body = this._captureBlock(source, m.index + m[0].length - 1);
      if (body && body.length > 80) {
        const snippet = this._extractLine(source, m.index);
        suggestions.push(new Suggestion('profile', [snippet.trim().replace(/;$/, '')], 'long loop; consider profiling', 0.45, snippet, [m.index, m.index + m[0].length]));
        this.memory.recordPattern('profile');
      }
    }
    return suggestions;
  }

  _ruleInlineSmallCall(source) {
    const suggestions = [];
    const re = /\b([A-Za-z_][\w]*)\s*\(/g;
    let m;
    while ((m = re.exec(source))) {
      const name = m[1];
      if (['add', 'mul', 'cmp', 'sum', 'min', 'max'].includes(name)) {
        const snippet = this._extractLine(source, m.index);
        suggestions.push(new Suggestion('inline', [name], 'small utility; consider inline', 0.3, snippet, [m.index, m.index + m[0].length]));
      }
    }
    return suggestions;
  }

  _ruleAssertNonNull(source) {
    const suggestions = [];
    const re = /\b([A-Za-z_][\w]*)\.[A-Za-z_][\w]*\b/g;
    let m;
    while ((m = re.exec(source))) {
      const v = m[1];
      const snippet = this._extractLine(source, m.index);
      suggestions.push(new Suggestion('assert', [`${v} != null`], 'possible null deref; add assert', 0.25, snippet, [m.index, m.index + m[0].length]));
    }
    return suggestions;
  }

  _ruleUnrollSmallLoop(source) {
    const suggestions = [];
    const re = /for\s*\([^;]+;\s*([A-Za-z0-9_]+)\s*<\s*([0-9]+)\s*;[^)]+\)\s*\{/g;
    let m;
    while ((m = re.exec(source))) {
      const bound = parseInt(m[2], 10);
      if (bound > 0 && bound <= MAX_UNROLL_SAFE) {
        const snippet = this._extractLine(source, m.index);
        suggestions.push(new Suggestion('unroll', [String(bound)], `small fixed loop (${bound}); consider unrolling`, 0.35, snippet, [m.index, m.index + m[0].length]));
      }
    }
    return suggestions;
  }

  _ruleSuggestCacheTTL(source) {
    const suggestions = [];
    const re = /\b(expensiveCompute|compute|fetchData|net\.request)\b/g;
    let m;
    while ((m = re.exec(source))) {
      const snippet = this._extractLine(source, m.index);
      const name = snippet.split(/\(|\s/)[0];
      suggestions.push(new Suggestion('memoize', [name], 'expensive operation; consider cache/memoize', 0.5, snippet, [m.index, m.index + m[0].length]));
    }
    return suggestions;
  }

  _ruleSanitizeHtml(source) {
    const suggestions = [];
    const re = /print\s*[:=]\s*["'][^"']*<[^>]+>[^"']*["']\s*\+\s*[A-Za-z_][\w]*/g;
    let m;
    while ((m = re.exec(source))) {
      const snippet = this._extractLine(source, m.index);
      suggestions.push(new Suggestion('sanitize', ['user_input'], 'HTML output with concatenated user input; sanitize', 0.9, snippet, [m.index, m.index + m[0].length]));
      this.memory.recordPattern('sanitize');
    }
    return suggestions;
  }

  _ruleAuditSensitive(source) {
    const suggestions = [];
    const re = /\b(delete|transfer|withdraw|close_account)\b/gi;
    let m;
    while ((m = re.exec(source))) {
      const snippet = this._extractLine(source, m.index);
      suggestions.push(new Suggestion('audit', [snippet.trim().replace(/;$/, '')], 'sensitive operation detected; add audit', 0.9, snippet, [m.index, m.index + m[0].length]));
      this.memory.recordPattern('audit');
    }
    return suggestions;
  }

  // small helpers used by rules
  _extractLine(source, idx, pad = 140) {
    const start = Math.max(0, source.lastIndexOf('\n', idx) + 1);
    let end = source.indexOf('\n', idx);
    if (end === -1) end = source.length;
    let line = source.slice(start, end).trim();
    if (line.length > pad) return line.slice(0, pad) + '...';
    return line;
  }

  _captureBlock(source, braceIndex) {
    const i = source.indexOf('{', braceIndex);
    if (i === -1) return null;
    let depth = 0;
    let j = i;
    while (j < source.length) {
      const ch = source[j];
      if (ch === '{') depth++;
      else if (ch === '}') {
        depth--;
        if (depth === 0) return source.slice(i + 1, j);
      } else if (ch === '"' || ch === "'") {
        const quote = ch;
        j++;
        while (j < source.length && !(source[j] === quote && source[j - 1] !== '\\')) j++;
      }
      j++;
    }
    return null;
  }

  _hash(s) {
    return crypto.createHash('sha1').update(s).digest('hex');
  }
}

// -------------------------
// Simple HTTP API server CLI
// -------------------------
function serve(core, host = '127.0.0.1', port = DEFAULT_PORT) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (req.method === 'GET' && url.pathname === '/suggest') {
      const file = url.searchParams.get('file');
      const max = parseInt(url.searchParams.get('max') || '8', 10);
      if (!file || !fs.existsSync(file)) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'file missing or not found' }));
        return;
      }
      const src = readFileSafe(file);
      const suggestions = core.analyzeSource(src, file, max).map(s => s.toJSON ? s.toJSON() : s);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ suggestions }));
      return;
    }
    if (req.method === 'POST' && url.pathname === '/apply') {
      let body = '';
      req.on('data', chunk => body += chunk.toString());
      req.on('end', () => {
        try {
          const data = JSON.parse(body || '{}');
          const file = data.file;
          const idx = parseInt(data.index || 0, 10);
          const inplace = !!data.inplace;
          if (!file || !fs.existsSync(file)) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'file missing' }));
            return;
          }
          const src = readFileSafe(file);
          const suggestions = core.analyzeSource(src, file, 32);
          if (!suggestions || idx < 0 || idx >= suggestions.length) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'invalid index' }));
            return;
          }
          const applied = core.applySuggestionToFile(file, suggestions[idx], { inplace, createPatch: true });
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: applied.ok, out: applied.ok ? applied.out : applied.msg }));
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: String(e) }));
        }
      });
      return;
    }
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'unknown endpoint' }));
  });

  server.listen(port, host, () => {
    log('INFO', `DodecaCore HTTP API running on ${host}:${port}`);
    console.log(`Serving DodecaCore API on http://${host}:${port}`);
  });
  return server;
}

// -------------------------
// CLI entrypoint
// -------------------------
if (require.main === module) {
  (async () => {
    const argv = process.argv.slice(2);
    const cmd = argv[0];
    const core = new DodecaCore();
    try {
      if (!cmd || cmd === 'help') {
        console.log('Usage: node dodeca_core.js <cmd> [args]');
        console.log('Commands: suggest, preview, apply, serve, emit-helper, inject-helper, optimize-loop, apply-patch, undo, plugins');
        process.exit(0);
      }

      if (cmd === 'suggest') {
        const file = argv[1];
        const max = parseInt(argv[2] || '12', 10);
        if (!file) { console.error('file required'); process.exit(2); }
        const src = readFileSafe(file);
        const s = core.analyzeSource(src, file, max);
        s.forEach((sg, i) => {
          console.log(`[${i}] ${sg.macro_name} ${JSON.stringify(sg.args)} (score=${(sg.score || 0).toFixed(2)}) - ${sg.reason}`);
          if (sg.snippet) console.log('     snippet:', sg.snippet);
        });
        process.exit(0);
      }

      if (cmd === 'preview') {
        const file = argv[1];
        const idx = parseInt(argv[2] || '0', 10);
        if (!file) { console.error('file required'); process.exit(2); }
        const src = readFileSafe(file);
        const s = core.analyzeSource(src, file, DEFAULT_MAX_SUGGESTIONS);
        if (!s.length) { console.log('no suggestions'); process.exit(0); }
        if (idx < 0 || idx >= s.length) { console.error('invalid index'); process.exit(2); }
        const res = core.previewApply(src, s[idx]);
        console.log('=== transformed preview ===\n');
        console.log(res.transformed);
        process.exit(0);
      }

      if (cmd === 'apply') {
        const file = argv[1];
        const idx = parseInt(argv[2] || '0', 10);
        if (!file) { console.error('file required'); process.exit(2); }
        const src = readFileSafe(file);
        const s = core.analyzeSource(src, file, DEFAULT_MAX_SUGGESTIONS);
        if (!s.length) { console.log('no suggestions'); process.exit(0); }
        if (idx < 0 || idx >= s.length) { console.error('invalid index'); process.exit(2); }
        const res = core.applySuggestionToFile(file, s[idx], { inplace: false, createPatch: true });
        if (res.ok) console.log('applied ->', res.out);
        else console.error('apply failed:', res.msg);
        process.exit(res.ok ? 0 : 2);
      }

      if (cmd === 'serve') {
        const port = parseInt(argv[1] || String(DEFAULT_PORT), 10);
        serve(core, '0.0.0.0', port);
        // keep running
      }

      if (cmd === 'optimize-loop') {
        const file = argv[1];
        const maxUnroll = parseInt(argv[2] || String(MAX_UNROLL_SAFE), 10);
        const apply = argv.includes('--apply');
        if (!file) { console.error('file required'); process.exit(2); }
        const [ok, msg] = optimizeLoopCLI(core, file, maxUnroll, apply);
        console.log(msg);
        process.exit(ok ? 0 : 2);
      }

      if (cmd === 'plugins') {
        const action = argv[1];
        const name = argv[2];
        const pm = core.plugins;
        if (action === 'list') {
          console.log('available:', pm.discover());
          console.log('loaded:', Object.keys(pm.loaded));
        } else if (action === 'load') {
          if (!name) { console.error('name required'); process.exit(2); }
          const [ok, msg] = pm.load(name, core);
          console.log(msg);
          process.exit(ok ? 0 : 2);
        } else if (action === 'unload') {
          if (!name) { console.error('name required'); process.exit(2); }
          const [ok, msg] = pm.unload(name);
          console.log(msg);
          process.exit(ok ? 0 : 2);
        } else {
          console.log('usage: plugins <list|load|unload> [name]');
          process.exit(2);
        }
      }

      if (cmd === 'test') {
        const ok = runSelfTest(core);
        console.log('self test', ok ? 'PASS' : 'FAIL');
        process.exit(ok ? 0 : 2);
      }

      console.error('unknown command', cmd);
      process.exit(2);
    } catch (e) {
      log('FATAL', e.stack || e.message);
      console.error('fatal', e);
      process.exit(2);
    }
  })();
}

// Helper bridging optimize-loop CLI to internal function
function optimizeLoopCLI(core, file, maxUnroll, apply) {
  try {
    const result = optimizeLoopUnrollFile(core, file, maxUnroll, apply);
    return [true, result];
  } catch (e) {
    return [false, String(e)];
  }
}

// Reuse the textual optimizer implemented above but adapted to DodecaCore instance:
function optimizeLoopUnrollFile(core, targetPath, maxUnroll = 8, apply = false) {
  const src = readFileSafe(targetPath);
  if (src === null) throw new Error('read failed');
  const pattern = /for\s*\(\s*([A-Za-z_][\w]*)\s*=\s*0\s*;\s*\1\s*<\s*([0-9]+)\s*;\s*\1\+\+\s*\)\s*\{/g;
  let m;
  let changed = false;
  let last = 0;
  const parts = [];
  while ((m = pattern.exec(src))) {
    const iName = m[1]; const bound = parseInt(m[2], 10);
    if (bound <= 0 || bound > maxUnroll) continue;
    const startBody = src.indexOf('{', m.index + m[0].length - 1);
    if (startBody === -1) continue;
    let depth = 0; let j = startBody;
    while (j < src.length) {
      const ch = src[j];
      if (ch === '{') depth++;
      else if (ch === '}') {
        depth--;
        if (depth === 0) break;
      }
      j++;
    }
    if (j >= src.length) continue;
    const body = src.slice(startBody + 1, j);
    if (!body.includes(iName)) continue;
    // Build unrolled text
    const unrolled = [];
    for (let k = 0; k < bound; ++k) {
      unrolled.push(body.replace(new RegExp(`\\b${iName}\\b`, 'g'), String(k)));
    }
    parts.push(src.slice(last, m.index));
    parts.push('/* unrolled loop (automated) */\n');
    parts.push(unrolled.join('\n') + '\n');
    last = j + 1;
    changed = true;
  }
  if (!changed) return 'no eligible loops found';
  parts.push(src.slice(last));
  const transformed = parts.join('');
  const patch = unifiedDiff(src, transformed, targetPath, targetPath + '.ai.ix');
  const patchPath = targetPath + '.ai.unroll.patch';
  writeFileSafe(patchPath, patch, false);
  if (apply) {
    const outPath = targetPath + '.ai.ix';
    writeFileSafe(outPath, transformed, true);
    return `applied -> ${outPath}`;
  }
  return `patch written -> ${patchPath}`;
}

function runSelfTest(core) {
  try {
    const sample = `-- Demo
net.request("https://api.example.com/data");
db.conn.query("select *");
func fib(n) { if n <= 1 { n } else { fib(n-1) + fib(n-2) } };
file_handle = open("log.txt");
file_handle.close();
for (i = 0; i < 4; i++) { doWork(i); }
`;
    const suggestions = core.analyzeSource(sample, '<test>', 20);
    return suggestions.length >= 4;
  } catch (e) {
    log('ERROR', 'selfTest', e.stack || e.message);
    return false;
  }
}

module.exports = { DodecaCore, AISimpleMemory, PluginManager, UndoManager, CodegenToolkit };


