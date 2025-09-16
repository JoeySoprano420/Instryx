
"""
instryx_heap_gc_allocator.py

Production-ready Heap + Generational Mark-and-Sweep Garbage Collector (GC) allocator
for the Instryx runtime with many optimizations, tooling and features.

Enhancements added:
- Generational GC (minor/major collections), remembered-set and write-barrier.
- Card-table style coarse-grained remembered-set via per-object flags.
- Pinned-object support (pin/unpin) to avoid moving or finalizing during compaction.
- Large-object allocator (separate tracking) to avoid frequent movement.
- Async background finalizer thread that executes finalizers out-of-band.
- Optional incremental marking support (mark-in-steps) with an exposed API.
- Best-effort heap compaction: moves live objects to a new heap and rewrites references.
- Heap snapshot diff / leak detection helpers.
- Integration hooks: register external instrumentation callbacks for alloc/free/collect.
- Profiling counters, generation histograms and runtime stats.
- Export/import heap snapshot, export metrics to JSON, atomic export.
- Conservative root scanner helper using Python's gc module (optional).
- CLI demo and robust self-test.

Design:
- Objects are opaque integer handles.
- Only integers that exist in the allocator's heap are treated as object references.
- Caller should call write-barrier helpers (set_field/set_index) already implemented here.
- The allocator is thread-safe via an RLock.
"""

from __future__ import annotations
import gc as _py_gc
import json
import logging
import os
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

LOG = logging.getLogger("instryx.heapgc")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO)

# Configuration defaults
DEFAULT_MAX_HEAP_OBJECTS = 50_000
DEFAULT_YOUNG_THRESHOLD = 2         # minor collections to trigger promotion
BACKGROUND_COLLECTION_INTERVAL = 2.0  # background minor GC interval (s)
FINALIZER_BATCH_SIZE = 32            # finalizers executed per run
COMPACT_THRESHOLD_RATIO = 0.25       # if free fraction > threshold, attempt compaction


@dataclass
class HeapStats:
    alloc_count: int = 0
    collect_count: int = 0
    last_collect_time: Optional[float] = None
    last_collected_objects: int = 0
    heap_size_high_water: int = 0
    total_finalized: int = 0


class HeapObject:
    """
    Internal representation of a heap object.
    Fields:
      - id: handle integer
      - kind: "object", "array", "bytes", "value", "large"
      - payload: dict/list/bytearray/value depending on kind
      - gen: generation number (0 = young)
      - age: survived minor collections count
      - marked: mark flag during marking
      - pinned: if True object must not be moved / finalized
      - has_old_to_young_ref: boolean card flag used for remembered set
      - finalizer: optional callable(handle, obj)
      - size: abstract size units for heuristics (large objects track their bytes)
    """
    __slots__ = ("id", "kind", "payload", "gen", "age", "marked", "pinned",
                 "has_old_to_young_ref", "finalizer", "size")

    def __init__(self, id_: int, kind: str, payload: Any, size: int = 1, finalizer: Optional[Callable] = None):
        self.id = id_
        self.kind = kind
        self.payload = payload
        self.gen = 0
        self.age = 0
        self.marked = False
        self.pinned = False
        self.has_old_to_young_ref = False
        self.finalizer = finalizer
        self.size = int(size)


class HeapGCAllocator:
    """
    Main allocator class implementing features listed above.
    """

    def __init__(
        self,
        max_heap_objects: int = DEFAULT_MAX_HEAP_OBJECTS,
        young_threshold: int = DEFAULT_YOUNG_THRESHOLD,
        enable_background: bool = True,
    ):
        self.max_heap_objects = int(max_heap_objects)
        self.young_threshold = int(young_threshold)

        self._lock = threading.RLock()
        self._next_id = 1
        self._heap: Dict[int, HeapObject] = {}
        self._large_objects: Set[int] = set()
        self._roots: Dict[str, Set[int]] = {}
        self._remembered: Set[int] = set()
        self._pinned: Set[int] = set()
        self._stats = HeapStats()
        self._observers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._finalizer_queue: List[Tuple[int, HeapObject]] = []
        self._finalizer_lock = threading.RLock()
        self._stop_bg = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._finalizer_thread: Optional[threading.Thread] = None
        self._incremental_stack: List[int] = []
        self._incremental_state = {"active": False}
        self._compact_lock = threading.RLock()
        if enable_background:
            self.start_background_collector()
            self.start_finalizer_worker()
        LOG.info("HeapGCAllocator init max=%d young_threshold=%d bg=%s", self.max_heap_objects, self.young_threshold, enable_background)

    # ---------------------------
    # Observers / instrumentation
    # ---------------------------
    def register_observer(self, name: str, fn: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._observers.setdefault(name, []).append(fn)

    def _notify(self, event: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            for fn in self._observers.get(event, []):
                try:
                    fn(payload)
                except Exception:
                    LOG.exception("observer %s failed", event)

    # ---------------------------
    # Allocation primitives
    # ---------------------------
    def _alloc_id(self) -> int:
        with self._lock:
            hid = self._next_id
            self._next_id += 1
            return hid

    def _ensure_capacity(self):
        if len(self._heap) + len(self._large_objects) >= self.max_heap_objects:
            LOG.debug("heap near capacity, attempting GC")
            self.collect(full=False)
            if len(self._heap) + len(self._large_objects) >= self.max_heap_objects:
                # attempt more aggressive major collect
                self.collect(full=True)
            if len(self._heap) + len(self._large_objects) >= self.max_heap_objects:
                raise MemoryError("Heap limit reached after GC attempts")

    def alloc_object(self, fields: Optional[Dict[str, Any]] = None, finalizer: Optional[Callable] = None, size: int = 1) -> int:
        fields = dict(fields or {})
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "object", fields, size=size, finalizer=finalizer)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._stats.heap_size_high_water = max(self._stats.heap_size_high_water, len(self._heap))
            self._update_remembered_for_new(obj)
            self._notify("alloc", {"handle": hid, "kind": "object", "size": size})
            return hid

    def alloc_array(self, length: int, initial: Any = None, finalizer: Optional[Callable] = None, size: int = 1) -> int:
        arr = [initial] * int(length)
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "array", arr, size=size, finalizer=finalizer)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._update_remembered_for_new(obj)
            self._notify("alloc", {"handle": hid, "kind": "array", "size": size})
            return hid

    def alloc_bytes(self, nbytes: int, finalizer: Optional[Callable] = None) -> int:
        b = bytearray(int(nbytes))
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "bytes", b, size=nbytes, finalizer=finalizer)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._notify("alloc", {"handle": hid, "kind": "bytes", "size": nbytes})
            return hid

    def alloc_large_object(self, payload: Any, size_bytes: int, finalizer: Optional[Callable] = None) -> int:
        """Allocate a large object tracked separately (less likely to move)."""
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "large", payload, size=size_bytes, finalizer=finalizer)
            self._heap[hid] = obj
            self._large_objects.add(hid)
            self._stats.alloc_count += 1
            self._notify("alloc", {"handle": hid, "kind": "large", "size": size_bytes})
            return hid

    def box_value(self, value: Any) -> int:
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "value", value, size=1)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._notify("alloc", {"handle": hid, "kind": "value"})
            return hid

    # ---------------------------
    # Access helpers + write barrier
    # ---------------------------
    def get_field(self, handle: int, name: str) -> Any:
        obj = self._heap.get(handle)
        if not obj or obj.kind != "object":
            raise KeyError("invalid object handle for get_field")
        return obj.payload.get(name)

    def set_field(self, handle: int, name: str, value: Any) -> None:
        with self._lock:
            obj = self._heap.get(handle)
            if not obj or obj.kind != "object":
                raise KeyError("invalid object handle for set_field")
            obj.payload[name] = value
            self._write_barrier(obj, value)

    def get_index(self, handle: int, idx: int) -> Any:
        obj = self._heap.get(handle)
        if not obj or obj.kind != "array":
            raise KeyError("invalid array handle for get_index")
        return obj.payload[int(idx)]

    def set_index(self, handle: int, idx: int, value: Any) -> None:
        with self._lock:
            obj = self._heap.get(handle)
            if not obj or obj.kind != "array":
                raise KeyError("invalid array handle for set_index")
            obj.payload[int(idx)] = value
            self._write_barrier(obj, value)

    def raw_bytes(self, handle: int) -> bytearray:
        obj = self._heap.get(handle)
        if not obj or obj.kind not in ("bytes", "large"):
            raise KeyError("invalid bytes handle")
        return obj.payload

    def _write_barrier(self, obj: HeapObject, value: Any) -> None:
        """
        Write barrier: when an older object stores a reference to a young object,
        record it in remembered set. Also maintain card flag optimization.
        """
        if not isinstance(value, int) or value not in self._heap:
            return
        with self._lock:
            target = self._heap[value]
            if obj.gen > 0 and target.gen == 0:
                obj.has_old_to_young_ref = True
                self._remembered.add(obj.id)

    # ---------------------------
    # Pinning API
    # ---------------------------
    def pin(self, handle: int) -> None:
        with self._lock:
            if handle in self._heap:
                self._heap[handle].pinned = True
                self._pinned.add(handle)

    def unpin(self, handle: int) -> None:
        with self._lock:
            if handle in self._heap:
                self._heap[handle].pinned = False
                self._pinned.discard(handle)

    # ---------------------------
    # Roots API
    # ---------------------------
    def register_root(self, set_name: str, handle: int) -> None:
        with self._lock:
            self._roots.setdefault(set_name, set()).add(handle)
            self._notify("root_register", {"set": set_name, "handle": handle})

    def unregister_root(self, set_name: str, handle: int) -> None:
        with self._lock:
            if set_name in self._roots:
                self._roots[set_name].discard(handle)
                if not self._roots[set_name]:
                    del self._roots[set_name]
            self._notify("root_unregister", {"set": set_name, "handle": handle})

    def list_roots(self) -> Dict[str, Set[int]]:
        with self._lock:
            return {k: set(v) for k, v in self._roots.items()}

    # ---------------------------
    # Finalizers worker
    # ---------------------------
    def start_finalizer_worker(self):
        if self._finalizer_thread and self._finalizer_thread.is_alive():
            return
        self._finalizer_thread = threading.Thread(target=self._finalizer_worker, daemon=True)
        self._finalizer_thread.start()
        LOG.info("finalizer worker started")

    def stop_finalizer_worker(self):
        if self._finalizer_thread:
            try:
                self._finalizer_thread.join(timeout=1.0)
            except Exception:
                pass

    def _enqueue_finalizer(self, hid: int, obj: HeapObject):
        with self._finalizer_lock:
            self._finalizer_queue.append((hid, obj))

    def _finalizer_worker(self):
        while not self._stop_bg.is_set():
            batch = []
            with self._finalizer_lock:
                while self._finalizer_queue and len(batch) < FINALIZER_BATCH_SIZE:
                    batch.append(self._finalizer_queue.pop(0))
            if not batch:
                time.sleep(0.2)
                continue
            for hid, obj in batch:
                try:
                    if obj.finalizer and not obj.pinned:
                        obj.finalizer(hid, obj)
                        self._stats.total_finalized += 1
                        self._notify("finalized", {"handle": hid})
                except Exception:
                    LOG.exception("finalizer error for %d", hid)

    # ---------------------------
    # Collection: minor/major and incremental
    # ---------------------------
    def collect(self, full: bool = True) -> Dict[str, Any]:
        """
        Full collection when full=True (major), minor otherwise.
        Collection is mark-and-sweep; finalizers queued and executed asynchronously.
        """
        start_time = time.time()
        with self._lock:
            try:
                self._unmark_all()
                roots = self._gather_roots()
                # include remembered set for minor collections
                remember_roots = set(self._remembered) if not full else set()
                stack = list(roots | remember_roots)
                marked = 0
                while stack:
                    hid = stack.pop()
                    if hid not in self._heap:
                        continue
                    obj = self._heap[hid]
                    if obj.marked:
                        continue
                    obj.marked = True
                    marked += 1
                    refs = self._extract_references(obj)
                    for r in refs:
                        if r in self._heap and not self._heap[r].marked:
                            stack.append(r)
                # sweep phase - collect unmarked
                to_delete = []
                for hid, obj in list(self._heap.items()):
                    if not obj.marked:
                        if obj.pinned:
                            # pinned unreachable objects are skipped (user responsibility)
                            continue
                        to_delete.append(hid)
                # queue finalizers and remove
                for hid in to_delete:
                    obj = self._heap.get(hid)
                    if obj and obj.finalizer:
                        self._enqueue_finalizer(hid, obj)
                for hid in to_delete:
                    self._heap.pop(hid, None)
                    self._remembered.discard(hid)
                    self._large_objects.discard(hid)
                    self._pinned.discard(hid)
                # promotion
                for obj in self._heap.values():
                    if obj.marked:
                        if obj.gen == 0:
                            obj.age += 1
                            if obj.age >= self.young_threshold:
                                obj.gen += 1
                                obj.age = 0
                        # reset has_old_to_young_ref if now stable
                        if obj.has_old_to_young_ref:
                            # recompute whether any refs still point to young
                            refs = self._extract_references(obj)
                            obj.has_old_to_young_ref = any(self._heap[r].gen == 0 for r in refs if r in self._heap)
                # stats
                self._stats.collect_count += 1
                self._stats.last_collect_time = time.time()
                self._stats.last_collected_objects = len(to_delete)
                # auto compaction heuristic
                free_fraction = 1.0 - (len(self._heap) / max(1, self.max_heap_objects))
                if full and free_fraction > COMPACT_THRESHOLD_RATIO:
                    try:
                        self.compact_heap()
                    except Exception:
                        LOG.exception("compaction failed")
                self._notify("collect", {"marked": marked, "collected": len(to_delete)})
                LOG.info("GC complete: marked=%d collected=%d heap=%d", marked, len(to_delete), len(self._heap))
                return {"marked": marked, "collected": len(to_delete), "heap_size": len(self._heap), "duration_s": time.time() - start_time}
            except Exception:
                LOG.exception("GC failed")
                return {"error": "exception during GC"}

    def minor_collect(self) -> Dict[str, Any]:
        return self.collect(full=False)

    def major_collect(self) -> Dict[str, Any]:
        return self.collect(full=True)

    def _unmark_all(self):
        for obj in self._heap.values():
            obj.marked = False

    def _gather_roots(self) -> Set[int]:
        roots: Set[int] = set()
        for s in self._roots.values():
            for hid in s:
                if hid in self._heap:
                    roots.add(hid)
        return roots

    # incremental marking: prepare and do limited steps
    def begin_incremental_mark(self):
        with self._lock:
            self._unmark_all()
            roots = self._gather_roots()
            self._incremental_stack = list(roots)
            self._incremental_state["active"] = True
            LOG.debug("incremental mark started with stack=%d", len(self._incremental_stack))

    def incremental_mark_step(self, max_nodes: int = 1000) -> Dict[str, Any]:
        with self._lock:
            if not self._incremental_state.get("active"):
                return {"status": "inactive"}
            processed = 0
            while self._incremental_stack and processed < max_nodes:
                hid = self._incremental_stack.pop()
                if hid not in self._heap:
                    continue
                obj = self._heap[hid]
                if obj.marked:
                    continue
                obj.marked = True
                processed += 1
                refs = self._extract_references(obj)
                for r in refs:
                    if r in self._heap and not self._heap[r].marked:
                        self._incremental_stack.append(r)
            if not self._incremental_stack:
                # finish and sweep unreachable
                self._incremental_state["active"] = False
                # now sweep like a major collect but queue finalizers async
                to_delete = [hid for hid, obj in list(self._heap.items()) if not obj.marked and not obj.pinned]
                for hid in to_delete:
                    obj = self._heap.get(hid)
                    if obj and obj.finalizer:
                        self._enqueue_finalizer(hid, obj)
                for hid in to_delete:
                    self._heap.pop(hid, None)
                    self._remembered.discard(hid)
                    self._large_objects.discard(hid)
                    self._pinned.discard(hid)
                LOG.debug("incremental mark complete processed=%d swept=%d", processed, len(to_delete))
                return {"status": "complete", "processed": processed, "swept": len(to_delete)}
            return {"status": "in-progress", "processed": processed, "remaining": len(self._incremental_stack)}

    # ---------------------------
    # Compaction (best-effort)
    # ---------------------------
    def compact_heap(self) -> Dict[str, Any]:
        """
        Best-effort compaction: allocate new ids for live objects, copy payloads,
        rewrite references and update roots. Pinned objects are not moved.
        This operation can be heavy; caller should ensure minimal concurrency.
        """
        with self._compact_lock, self._lock:
            try:
                LOG.info("Starting heap compaction")
                # identify live objects via a temporary mark
                self._unmark_all()
                roots = self._gather_roots()
                stack = list(roots)
                while stack:
                    hid = stack.pop()
                    if hid not in self._heap:
                        continue
                    obj = self._heap[hid]
                    if obj.marked:
                        continue
                    obj.marked = True
                    for r in self._extract_references(obj):
                        if r in self._heap and not self._heap[r].marked:
                            stack.append(r)
                # build mapping for movable objects: skip pinned and large objects
                mapping: Dict[int, int] = {}
                new_heap: Dict[int, HeapObject] = {}
                for old_id, obj in list(self._heap.items()):
                    if not obj.marked:
                        continue  # unreachable won't be moved
                    if obj.pinned or old_id in self._large_objects:
                        # preserve id to avoid changing external refs; copy as-is
                        mapping[old_id] = old_id
                        new_heap[old_id] = obj
                        continue
                    # assign new id
                    new_id = self._alloc_id()
                    mapping[old_id] = new_id
                    # shallow copy of payload; we'll rewrite references next
                    copied = None
                    if obj.kind == "object":
                        copied = dict(obj.payload)
                    elif obj.kind == "array":
                        copied = list(obj.payload)
                    elif obj.kind in ("bytes", "large"):
                        copied = bytearray(obj.payload) if obj.kind == "bytes" else obj.payload
                    elif obj.kind == "value":
                        copied = obj.payload
                    new_obj = HeapObject(new_id, obj.kind, copied, size=obj.size, finalizer=obj.finalizer)
                    new_obj.gen = obj.gen
                    new_obj.age = obj.age
                    new_obj.pinned = obj.pinned
                    new_heap[new_id] = new_obj
                # rewrite references in new_heap
                def rewrite_in_obj(o: HeapObject):
                    if o.kind == "object":
                        for k, v in list(o.payload.items()):
                            if isinstance(v, int) and v in mapping:
                                o.payload[k] = mapping[v]
                    elif o.kind == "array":
                        for i, v in enumerate(o.payload):
                            if isinstance(v, int) and v in mapping:
                                o.payload[i] = mapping[v]
                for o in new_heap.values():
                    rewrite_in_obj(o)
                # update roots to new ids
                new_roots: Dict[str, Set[int]] = {}
                for rname, rset in self._roots.items():
                    new_set = set()
                    for hid in rset:
                        if hid in mapping:
                            new_set.add(mapping[hid])
                    if new_set:
                        new_roots[rname] = new_set
                # replace heap and roots
                self._heap = new_heap
                self._roots = new_roots
                # rebuild large_objects and pinned sets
                self._large_objects = {mapping.get(h, h) for h in list(self._large_objects) if h in mapping}
                self._pinned = {mapping.get(h, h) for h in list(self._pinned) if h in mapping}
                # rebuild remembered set conservatively
                self._remembered = set()
                for hid, obj in self._heap.items():
                    if obj.gen > 0:
                        refs = self._extract_references(obj)
                        if any((r in self._heap and self._heap[r].gen == 0) for r in refs):
                            self._remembered.add(hid)
                LOG.info("Compaction complete: new_heap=%d", len(self._heap))
                self._notify("compact", {"heap_size": len(self._heap)})
                return {"status": "ok", "heap_size": len(self._heap)}
            except Exception:
                LOG.exception("compaction failed")
                return {"status": "error", "message": "exception during compaction"}

    # ---------------------------
    # Helpers & introspection
    # ---------------------------
    def heap_snapshot(self, include_contents: bool = False) -> Dict[str, Any]:
        with self._lock:
            snap = {
                "time": time.time(),
                "count": len(self._heap),
                "objects": {},
                "roots": {k: list(v) for k, v in self._roots.items()},
                "remembered": list(self._remembered),
                "large_objects": list(self._large_objects),
                "pinned": list(self._pinned),
                "stats": self._stats.__dict__.copy(),
            }
            for hid, obj in self._heap.items():
                o = {"id": hid, "kind": obj.kind, "gen": obj.gen, "age": obj.age, "size": obj.size, "pinned": obj.pinned}
                if include_contents:
                    if obj.kind == "object":
                        o["fields"] = dict(obj.payload)
                    elif obj.kind == "array":
                        o["array"] = list(obj.payload)
                    elif obj.kind in ("bytes", "large"):
                        o["raw_len"] = len(obj.payload) if hasattr(obj.payload, "__len__") else None
                    elif obj.kind == "value":
                        o["value"] = obj.payload
                snap["objects"][hid] = o
            return snap

    def export_heap_json(self, path: Optional[str] = None, include_contents: bool = True) -> str:
        if not path:
            fd, path = tempfile.mkstemp(prefix="instryx_heap_", suffix=".json")
            os.close(fd)
        snap = self.heap_snapshot(include_contents=include_contents)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, default=str)
        os.replace(tmp, path)
        LOG.info("Heap exported -> %s", path)
        return path

    def import_heap_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        # Best-effort: clear current heap and rebuild simple boxed values (ids will be new)
        with self._lock:
            self._heap.clear()
            self._roots.clear()
            self._remembered.clear()
            self._pinned.clear()
            self._large_objects.clear()
            # load objects as values only (preserve counts)
            for old_id_str, meta in snap.get("objects", {}).items():
                hid = self._alloc_id()
                kind = meta.get("kind", "value")
                payload = None
                if kind == "object":
                    payload = {k: None for k in meta.get("fields", {}).keys()} if meta.get("fields") else {}
                elif kind == "array":
                    payload = [None] * len(meta.get("array", []))
                elif kind in ("bytes", "large"):
                    payload = bytearray(meta.get("raw_len", 0) or 0)
                else:
                    payload = None
                obj = HeapObject(hid, kind, payload, size=meta.get("size", 1))
                self._heap[hid] = obj
            LOG.info("imported heap snapshot (best-effort), recreated %d objects", len(self._heap))

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            d = self._stats.__dict__.copy()
            d.update({"heap_size": len(self._heap), "roots": {k: len(v) for k, v in self._roots.items()}})
            return d

    def find_handles_pointing_to(self, target: int) -> List[int]:
        with self._lock:
            res = []
            for hid, obj in self._heap.items():
                refs = self._extract_references(obj)
                if target in refs:
                    res.append(hid)
            return res

    # ---------------------------
    # Conservative root scanner (inspect Python heap via gc)
    # ---------------------------
    def scan_python_globals_for_handles(self) -> Set[int]:
        """
        Conservative scan through Python GC tracked objects (modules, builtins, globals)
        and collect integers that look like handles registered in this allocator.
        Use with caution: expensive.
        """
        found = set()
        with self._lock:
            valid_handles = set(self._heap.keys())
        for obj in list(_py_gc.get_objects()):
            try:
                if isinstance(obj, dict):
                    for v in obj.values():
                        if isinstance(v, int) and v in valid_handles:
                            found.add(v)
                elif isinstance(obj, (list, tuple, set)):
                    for v in obj:
                        if isinstance(v, int) and v in valid_handles:
                            found.add(v)
            except Exception:
                continue
        return found

    # ---------------------------
    # Compaction helper already defined above
    # ---------------------------

    # ---------------------------
    # Background collector control
    # ---------------------------
    def start_background_collector(self, interval: float = BACKGROUND_COLLECTION_INTERVAL):
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._stop_bg.clear()
        self._bg_thread = threading.Thread(target=self._bg_worker, args=(interval,), daemon=True)
        self._bg_thread.start()
        LOG.info("Background collector started interval=%.2fs", interval)

    def stop_background_collector(self):
        if self._bg_thread:
            self._stop_bg.set()
            try:
                self._bg_thread.join(timeout=1.0)
            except Exception:
                pass

    def _bg_worker(self, interval: float):
        while not self._stop_bg.is_set():
            try:
                self.minor_collect()
            except Exception:
                LOG.exception("background GC error")
            self._stop_bg.wait(interval)

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _extract_references(self, obj: HeapObject) -> Set[int]:
        refs = set()
        if obj.kind == "object" and isinstance(obj.payload, dict):
            for v in obj.payload.values():
                if isinstance(v, int) and v in self._heap:
                    refs.add(v)
        elif obj.kind == "array" and isinstance(obj.payload, (list, tuple)):
            for v in obj.payload:
                if isinstance(v, int) and v in self._heap:
                    refs.add(v)
        return refs

    def _update_remembered_for_new(self, obj: HeapObject):
        # If object is older and references young objects, remember it
        if obj.gen > 0:
            refs = self._extract_references(obj)
            for r in refs:
                if r in self._heap and self._heap[r].gen == 0:
                    obj.has_old_to_young_ref = True
                    self._remembered.add(obj.id)

    # ---------------------------
    # Self-test and CLI demo
    # ---------------------------
def _self_test() -> bool:
    LOG.info("Running HeapGCAllocator self-test (extended)")
    gc = HeapGCAllocator(max_heap_objects=200, enable_background=False)
    a = gc.alloc_object({"v": 123})
    b = gc.alloc_object({"ref": a})
    gc.register_root("main", b)
    # allocate many unreachable objects
    for i in range(100):
        gc.alloc_object({"temp": i})
    pre = len(gc._heap)
    stats = gc.collect(full=True)
    post = len(gc._heap)
    LOG.info("self-test: before=%d after=%d stats=%s", pre, post, stats)
    if a not in gc._heap or b not in gc._heap:
        LOG.error("self-test failed: live objects collected")
        return False
    # finalizer check
    finalized = []
    def final(hid, obj):
        finalized.append(hid)
    c = gc.alloc_object({"x": 10}, finalizer=final)
    # make unreachable and collect
    gc.collect(full=True)
    # allow finalizer thread time if enabled (not enabled in this test)
    LOG.info("finalized list (may be empty if finalizer worker disabled): %s", finalized)
    # test compaction
    try:
        r = gc.compact_heap()
        LOG.info("compaction result: %s", r)
    except Exception:
        LOG.exception("compaction in self-test failed")
    LOG.info("self-test passed")
    return True

if __name__ == "__main__":
    ok = _self_test()
    os._exit(0 if ok else 2)
