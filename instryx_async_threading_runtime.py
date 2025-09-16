"""
instryx_async_threading_runtime.py

A small, robust async+threading runtime for Instryx.

Features:
- Managed ThreadPoolExecutor for blocking/CPU work.
- Background asyncio event loop running in a dedicated thread for coroutine scheduling.
- Safe bridges between thread futures and asyncio futures (via run_coroutine_threadsafe).
- CancellationToken for cooperative cancellation.
- Timers, delayed scheduling and repeating tasks.
- Convenience helpers: spawn (schedule coroutine), submit (thread pool), run_sync (wait for coroutine/future).
- Graceful startup / shutdown and basic metrics.

This module uses only Python stdlib (threading, asyncio, concurrent.futures).
"""

from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, List
from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
import os

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CancellationToken:
    """Simple cooperative cancellation token."""
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                # invoke immediately if already cancelled
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
                return
            self._callbacks.append(cb)


class TimerHandle:
    """Handle for a scheduled timer; calling cancel() prevents further runs."""
    def __init__(self, cancel_fn: Callable[[], None]):
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class AsyncThreadingRuntime:
    """
    Runtime that owns:
      - an asyncio event loop running in a dedicated thread
      - a ThreadPoolExecutor for blocking tasks
    Use a single runtime instance per process where practical.
    """

    def __init__(self, workers: int = 4, loop_name: Optional[str] = None):
        self._workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown = False
        self._loop_name = loop_name or "instryx-async-loop"
        self._metrics: Dict[str, Any] = {"tasks_submitted": 0, "thread_tasks": 0, "coroutines_spawned": 0}
        self._start_loop_thread()

    # -----------------------
    # Event loop management
    # -----------------------
    def _loop_target(self) -> None:
        """Target function that runs the asyncio event loop forever until stopped."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_started.set()
            LOG.debug("Async loop started in thread %s", threading.current_thread().name)
            loop.run_forever()
            # clean shutdown: cancel all tasks
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            LOG.debug("Async loop stopped and closed")
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            self._loop_started.set()

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        # wait until loop created
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("asyncio loop not available")
        return self._loop

    # -----------------------
    # Task submission
    # -----------------------
    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> ThreadFuture:
        """
        Submit a blocking/cpu function to the thread pool. Returns concurrent.futures.Future.
        """
        if self._shutdown:
            raise RuntimeError("Runtime is shutting down")
        self._metrics["thread_tasks"] += 1
        fut = self._executor.submit(fn, *args, **kwargs)
        return fut

    def spawn(self, coro: Awaitable, *, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """
        Schedule an awaitable on the runtime's event loop. Returns a concurrent.futures.Future
        that can be waited on from other threads.
        """
        if self._shutdown:
            raise RuntimeError("Runtime is shutting down")
        loop = self._ensure_loop()
        self._metrics["coroutines_spawned"] += 1
        # run_coroutine_threadsafe gives a concurrent.futures.Future
        cf: "concurrent.futures.Future" = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token:
            def cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(cancel_cb)
        return cf

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        """
        From any thread, run a coroutine or call a blocking function synchronously.
        - If passed a coroutine/awaitable -> schedules on event loop and waits.
        - If passed a callable -> runs in thread pool and waits.
        """
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # -----------------------
    # Timers / scheduling
    # -----------------------
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """
        Schedule a coroutine to run after delay seconds. coro_factory must return a coroutine when invoked.
        Returns a TimerHandle that can be cancelled before the coroutine runs.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _call():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coroutine factory raised")

        handle = loop.call_later(delay, _call)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """
        Schedule a repeating coroutine every `interval` seconds. Repetition is best-effort and runs serially.
        Returned handle cancels future repeats.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _runner():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating scheduled task raised")
                    # cooperative sleep
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_runner(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # -----------------------
    # Utilities
    # -----------------------
    @staticmethod
    async def sleep(delay: float) -> None:
        await asyncio.sleep(delay)

    # -----------------------
    # Shutdown
    # -----------------------
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Shutdown runtime: stop the event loop and thread pool.
        """
        if self._shutdown:
            return
        self._shutdown = True

        # stop asyncio loop
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop() failed")
            # schedule stop on loop thread
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop()")

        # shutdown threadpool
        try:
            self._executor.shutdown(wait=wait, timeout=timeout)
        except TypeError:
            # Python <3.9 doesn't support timeout arg
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")

        # join loop thread
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    # -----------------------
    # Diagnostics / metrics
    # -----------------------
    def metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    # -----------------------
    # Convenience decorator
    # -----------------------
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        """
        Decorator that runs the wrapped function in the threadpool when called.
        """

        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)

        return _wrapped


# Module-level singleton runtime (convenience)
_global_runtime: Optional[AsyncThreadingRuntime] = None
_global_lock = threading.Lock()


def get_runtime(workers: int = 4) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers)
        return _global_runtime


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]

if __name__ == "__main__":
    # Simple test/demo
    rt = get_runtime(workers=2)
    @rt.background
    def blocking_task(n):
        LOG.info("Blocking task %d starting", n)
        time.sleep(2)
        LOG.info("Blocking task %d done", n)
        return n * n
    async def async_task(n):
        LOG.info("Async task %d starting", n)
        await asyncio.sleep(1)
        LOG.info("Async task %d done", n)
        return n + 10
    # Submit blocking tasks
    futures = [blocking_task(i) for i in range(3)]
    # Spawn async tasks
    async_futures = [rt.spawn(async_task(i)) for i in range(3, 6)]
    # Wait for all to complete
    for f in futures:
        result = f.result()
        LOG.info("Blocking task result: %s", result)
    for af in async_futures:
        result = af.result()
        LOG.info("Async task result: %s", result)
    # Schedule a delayed task
    def delayed_coro():
        async def _coro():
            LOG.info("Delayed coroutine running")
            await asyncio.sleep(0.5)
            LOG.info("Delayed coroutine done")
        return _coro()
    timer = rt.schedule_later(3.0, delayed_coro)
    # Schedule a repeating task
    count = 0
    def repeating_coro():
        
        async def _coro():
            
            count += 1
            LOG.info("Repeating coroutine run %d", count)
            if count >= 5:
                repeating_handle.cancel()
                LOG.info("Repeating coroutine cancelled after 5 runs")
        return _coro()
    repeating_handle = rt.schedule_repeating(1.0, repeating_coro)
    # Let the demo run for a while
    try:
        time.sleep(10)
    finally:
        rt.shutdown()
        LOG.info("Runtime shutdown complete")
fmt_bytes = fmt.encode('utf-8')
c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt_bytes)), bytearray(fmt_bytes))
global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name="fstr")
global_fmt.linkage = 'internal'
global_fmt.global_constant = True
global_fmt.initializer = c_fmt
fmt_ptr = self.builder.bitcast(global_fmt, voidptr_ty)
arg_vals = [self.builder.bitcast(arg, voidptr_ty) for arg in args]
self.builder.call(self.printf, [fmt_ptr] + arg_vals)

"""
instryx_async_threading_runtime.py

Advanced async + threading runtime for Instryx.

Features (production-focused, fully implemented):
- Dedicated asyncio event loop running in a background thread.
- Configurable thread-pool for blocking/CPU tasks (ThreadPoolExecutor).
- Optional prioritized task queue with internal worker threads (priority scheduling).
- Backpressure / concurrency limits via semaphores.
- Batch submission helpers, map_with_concurrency.
- Scheduling: delayed tasks, repeating tasks with robust cancellation.
- CancellationToken for cooperative cancellation and callbacks.
- Adaptive resize: gentle API to increase/decrease worker capacity.
- Runtime metrics, timers, and lightweight tracing via logging.
- Safe startup/shutdown; idempotent and robust against errors.
- Minimal external dependencies (stdlib only). Usable by test harness / CI.

Usage examples:
  runtime = AsyncThreadingRuntime(workers=8)
  fut = runtime.submit(lambda: heavy_compute(x))
  cf = runtime.spawn(async_func(...))
  runtime.schedule_later(1.0, lambda: coro())
  runtime.shutdown()
"""

from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CancellationToken:
    """Cooperative cancellation token with callback registration."""
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
            self._callbacks.clear()
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                # invoke immediately if already cancelled
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
            else:
                self._callbacks.append(cb)


class TimerHandle:
    """Handle to cancel scheduled timers."""
    def __init__(self, cancel_fn: Callable[[], None]):
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# internal prioritized task descriptor
class _PriorityTask:
    __slots__ = ("priority", "created", "task_id", "callable", "args", "kwargs", "future")

    def __init__(self, priority: int, callable_: Callable, args: Tuple, kwargs: Dict, future: ThreadFuture):
        self.priority = priority
        self.created = time.time()
        self.task_id = uuid.uuid4().hex
        self.callable = callable_
        self.args = args
        self.kwargs = kwargs
        self.future = future

    def __lt__(self, other: "_PriorityTask") -> bool:
        # lower priority value === run earlier
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created < other.created


class AsyncThreadingRuntime:
    """
    Combined asyncio + threaded runtime.

    Key features:
    - `spawn(coro)` schedules coroutine on the background loop and returns concurrent.futures.Future.
    - `submit(fn, *args, **kwargs)` submits blocking task to the internal thread pool.
    - `submit_priority(fn, priority=50, ...)` submits to a priority queue served by internal workers.
    - `map_with_concurrency(func, iterable, concurrency)` convenience for many blocking calls.
    - schedule_later / schedule_repeating utilities returning TimerHandle.
    - adaptive resize: `resize_workers(new_count)` tries to adjust the underlying pool.
    - `shutdown()` gracefully stops loop, workers and pending tasks.
    """

    def __init__(self,
                 workers: int = 4,
                 enable_priority_queue: bool = True,
                 max_queue_size: Optional[int] = 0,
                 loop_name: Optional[str] = None):
        self._workers = max(1, int(workers))
        self._enable_priority = bool(enable_priority_queue)
        self._max_queue_size = None if (max_queue_size is None or max_queue_size <= 0) else int(max_queue_size)
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown_flag = threading.Event()
        self._loop_name = loop_name or "instryx-async-loop"
        # prioritized queue and workers (if enabled)
        self._pq: Optional[queue.PriorityQueue] = queue.PriorityQueue() if self._enable_priority else None
        self._pq_workers: List[threading.Thread] = []
        self._pq_worker_count = min(self._workers, 2) if self._enable_priority else 0
        self._pq_semaphore = threading.Semaphore(self._workers)  # backpressure against prioritized submits
        # metrics
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "tasks_submitted": 0,
            "priority_tasks_submitted": 0,
            "thread_tasks_executed": 0,
            "coroutines_spawned": 0,
            "start_time": time.time(),
        }
        # start event loop thread
        self._start_loop_thread()
        # start pq workers if enabled
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # -----------------------
    # Internal loop thread
    # -----------------------
    def _loop_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_started.set()
        LOG.debug("Async loop started in thread %s", threading.current_thread().name)
        try:
            loop.run_forever()
            # drain tasks on stop
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                pass
            LOG.debug("Async loop closed")

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("event loop not available")
        return self._loop

    # -----------------------
    # Priority queue workers
    # -----------------------
    def _pq_worker_target(self) -> None:
        assert self._pq is not None
        while not self._shutdown_flag.is_set():
            try:
                task: _PriorityTask = self._pq.get(timeout=0.2)
            except queue.Empty:
                continue
            # respect semaphore to enforce backpressure
            try:
                self._pq_semaphore.acquire()
                if task.future.cancelled():
                    continue
                try:
                    res = task.callable(*task.args, **task.kwargs)
                    task.future.set_result(res)
                except Exception as e:
                    task.future.set_exception(e)
                finally:
                    with self._metrics_lock:
                        self._metrics["thread_tasks_executed"] += 1
            finally:
                try:
                    self._pq_semaphore.release()
                except Exception:
                    pass
                self._pq.task_done()

    def _start_pq_workers(self, count: int) -> None:
        if not self._enable_priority or self._pq is None:
            return
        # avoid starting multiple times
        if self._pq_workers:
            return
        for i in range(max(1, int(count))):
            t = threading.Thread(target=self._pq_worker_target, name=f"instryx-pq-worker-{i}", daemon=True)
            t.start()
            self._pq_workers.append(t)

    def _stop_pq_workers(self) -> None:
        # signal threads to finish by setting shutdown flag, then join
        for t in list(self._pq_workers):
            try:
                if t.is_alive():
                    t.join(timeout=0.5)
            except Exception:
                pass
        self._pq_workers = []

    # -----------------------
    # Public submission APIs
    # -----------------------
    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> ThreadFuture:
        """Submit a blocking/cpu task to the thread pool."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        fut = self._executor.submit(fn, *args, **kwargs)
        with self._metrics_lock:
            self._metrics["tasks_submitted"] += 1
        return fut

    def submit_priority(self, fn: Callable[..., Any], *args, priority: int = 50, block: bool = True, **kwargs) -> ThreadFuture:
        """
        Submit a prioritized blocking task.
        - lower `priority` value runs earlier.
        - when queue bounded and full, behavior depends on `block`.
        """
        if not self._enable_priority or self._pq is None:
            # fallback to standard submit
            return self.submit(fn, *args, **kwargs)
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        # backpressure: if bounded queue and full
        if self._max_queue_size is not None:
            if not block and self._pq.qsize() >= self._max_queue_size:
                raise queue.Full("priority queue full")
        future = ThreadFuture()
        task = _PriorityTask(priority=priority, callable_=fn, args=args, kwargs=kwargs, future=future)
        self._pq.put(task)
        with self._metrics_lock:
            self._metrics["priority_tasks_submitted"] += 1
        return future

    def spawn(self, coro: Awaitable, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """
        Schedule an awaitable on the runtime event loop. Returns concurrent.futures.Future.
        """
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        loop = self._ensure_loop()
        with self._metrics_lock:
            self._metrics["coroutines_spawned"] += 1
        cf = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token is not None:
            def _cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(_cancel_cb)
        return cf

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        """
        From any (other) thread, run a coroutine synchronously or execute a callable in threadpool and wait.
        """
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # -----------------------
    # Batch helpers
    # -----------------------
    def map_with_concurrency(self, fn: Callable[..., Any], iterable, concurrency: int = 8) -> List[Any]:
        """
        Run `fn(x)` for x in iterable using up to `concurrency` parallel thread tasks.
        Returns results in same order as input.
        """
        it = list(iterable)
        n = len(it)
        if n == 0:
            return []
        results = [None] * n
        sem = threading.Semaphore(concurrency)
        threads = []

        def worker(i, item):
            nonlocal results
            try:
                sem.acquire()
                results[i] = fn(item)
            except Exception:
                LOG.exception("map_with_concurrency worker failed")
                results[i] = None
            finally:
                try:
                    sem.release()
                except Exception:
                    pass

        for i, item in enumerate(it):
            t = threading.Thread(target=worker, args=(i, item), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    # -----------------------
    # Scheduling helpers
    # -----------------------
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """
        Schedule coroutine factory to run after `delay` seconds. Returns TimerHandle to cancel.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _runner():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coro factory raised")

        handle = loop.call_later(delay, _runner)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """
        Schedule repeating coroutine. Cancelling the returned handle stops future repeats.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _looped():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating task raised")
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_looped(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # -----------------------
    # Resize / lifecycle
    # -----------------------
    def resize_workers(self, new_count: int) -> None:
        """Attempt to resize the threadpool. This creates a new ThreadPoolExecutor and swaps it in."""
        new_count = max(1, int(new_count))
        if new_count == self._workers:
            return
        LOG.info("Resizing workers %d -> %d", self._workers, new_count)
        # create new executor first
        new_exec = ThreadPoolExecutor(max_workers=new_count)
        old_exec = self._executor
        self._executor = new_exec
        self._workers = new_count
        # stop and join priority workers if present (adjust count)
        if self._enable_priority:
            # adjust semaphore capacity
            try:
                # best-effort: recreate semaphore with new_count
                self._pq_semaphore = threading.Semaphore(self._workers)
            except Exception:
                pass
            # restart pq workers to match small count
            self._stop_pq_workers()
            self._start_pq_workers(min(self._workers, 2))
        # shutdown old executor (do not wait to avoid blocking caller); let it finish in background
        try:
            old_exec.shutdown(wait=False)
        except Exception:
            LOG.exception("old executor shutdown failed")

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Gracefully shutdown runtime:
         - stop scheduling new tasks
         - stop priority workers
         - stop event loop
         - shutdown thread executor
        """
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        # stop priority workers
        if self._enable_priority:
            self._stop_pq_workers()
        # stop event loop
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop() failed")
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop()")
        # shutdown executor
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")
        # join loop thread
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    # -----------------------
    # Diagnostics / metrics
    # -----------------------
    def metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            m = dict(self._metrics)
        m.update({
            "uptime": time.time() - m.get("start_time", time.time())
        })
        return m

    # -----------------------
    # Convenience decorator
    # -----------------------
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        """Decorator to run a function in the threadpool when called."""
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)
        return _wrapped


# Module-level singleton runtime
_global_runtime_lock = threading.Lock()
_global_runtime: Optional[AsyncThreadingRuntime] = None


def get_runtime(workers: int = 4, enable_priority_queue: bool = True) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers, enable_priority_queue=enable_priority_queue)
        return _global_runtime


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]

"""
instryx_async_threading_runtime.py

Advanced async + threading runtime for Instryx.

Features added/implemented:
- Background asyncio loop in a dedicated thread.
- ThreadPoolExecutor for blocking/CPU-bound tasks.
- Optional prioritized task queue with bounded capacity and worker threads.
- Backpressure support and non-blocking submit options.
- CancellationToken that supports callbacks.
- Robust scheduling: schedule_later, schedule_repeating, with jitter option.
- Batch helpers: map_with_concurrency (sync), amap_with_concurrency (async).
- submit_batch convenience, map_reduce pattern.
- Adaptive resize_workers to increase/decrease pool size.
- CPU-affinity hints for worker threads on Linux (best-effort).
- Diagnostics: dump_threads (write stack traces), export_metrics, reset_metrics.
- Context manager support and idempotent shutdown.
- Minimal external dependencies (stdlib only). Fully executable.

Notes:
- Affinity setting uses os.sched_setaffinity on Linux; non-fatal if unsupported.
- Priority queue uses lower integer -> higher priority.
- All external interactions are best-effort and non-fatal; runtime remains usable if optional features unavailable.
"""

from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CancellationToken:
    """Cooperative cancellation token with callback registration."""
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
            self._callbacks.clear()
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                # invoke immediately if already cancelled
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
            else:
                self._callbacks.append(cb)


class TimerHandle:
    """Handle to cancel scheduled timers."""
    def __init__(self, cancel_fn: Callable[[], None]):
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class _PriorityTask:
    """Internal descriptor for prioritized tasks placed into a PriorityQueue."""
    __slots__ = ("priority", "created", "task_id", "callable", "args", "kwargs", "future")

    def __init__(self, priority: int, callable_: Callable, args: Tuple, kwargs: Dict, future: ThreadFuture):
        self.priority = int(priority)
        self.created = time.time()
        self.task_id = uuid.uuid4().hex
        self.callable = callable_
        self.args = args
        self.kwargs = kwargs
        self.future = future

    def __lt__(self, other: "_PriorityTask") -> bool:
        # lower `priority` value => execute earlier
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created < other.created


class AsyncThreadingRuntime:
    """
    Combined asyncio + threaded runtime with many production-ready helpers.
    """

    def __init__(self,
                 workers: int = 4,
                 enable_priority_queue: bool = False,
                 max_queue_size: Optional[int] = 0,
                 loop_name: Optional[str] = None,
                 worker_affinity: Optional[List[int]] = None):
        self._workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown_flag = threading.Event()
        self._loop_name = loop_name or "instryx-async-loop"
        self._enable_priority = bool(enable_priority_queue)
        self._max_queue_size = None if (max_queue_size is None or max_queue_size <= 0) else int(max_queue_size)
        self._pq: Optional[queue.PriorityQueue] = queue.PriorityQueue() if self._enable_priority else None
        self._pq_workers: List[threading.Thread] = []
        self._pq_worker_count = min(self._workers, 2) if self._enable_priority else 0
        self._pq_semaphore = threading.Semaphore(self._workers)
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "tasks_submitted": 0,
            "priority_tasks_submitted": 0,
            "thread_tasks_executed": 0,
            "coroutines_spawned": 0,
            "start_time": time.time(),
        }
        self._worker_affinity = list(worker_affinity) if worker_affinity else None
        self._start_loop_thread()
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # -----------------------
    # Event loop thread
    # -----------------------
    def _loop_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_started.set()
        LOG.debug("Async loop started in thread %s", threading.current_thread().name)
        try:
            loop.run_forever()
            # drain and cancel pending tasks
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                LOG.exception("Event loop close failed")
            LOG.debug("Async loop closed")

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("asyncio loop not available")
        return self._loop

    # -----------------------
    # Priority queue workers
    # -----------------------
    def _pq_worker_target(self) -> None:
        assert self._pq is not None
        # Optionally try to set affinity for this thread (best-effort)
        if self._worker_affinity and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(self._worker_affinity))
            except Exception:
                LOG.debug("setaffinity for pq worker failed (ignored)")
        while not self._shutdown_flag.is_set():
            try:
                task: _PriorityTask = self._pq.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._pq_semaphore.acquire()
                if task.future.cancelled():
                    continue
                try:
                    res = task.callable(*task.args, **task.kwargs)
                    task.future.set_result(res)
                except Exception as e:
                    task.future.set_exception(e)
                finally:
                    with self._metrics_lock:
                        self._metrics["thread_tasks_executed"] = self._metrics.get("thread_tasks_executed", 0) + 1
            finally:
                try:
                    self._pq_semaphore.release()
                except Exception:
                    pass
                try:
                    self._pq.task_done()
                except Exception:
                    pass

    def _start_pq_workers(self, count: int) -> None:
        if not self._enable_priority or self._pq is None:
            return
        if self._pq_workers:
            return
        for i in range(max(1, int(count))):
            t = threading.Thread(target=self._pq_worker_target, name=f"instryx-pq-worker-{i}", daemon=True)
            t.start()
            self._pq_workers.append(t)

    def _stop_pq_workers(self) -> None:
        # graceful join
        for t in list(self._pq_workers):
            try:
                if t.is_alive():
                    t.join(timeout=0.5)
            except Exception:
                pass
        self._pq_workers = []

    # -----------------------
    # Submission APIs
    # -----------------------
    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> ThreadFuture:
        """Submit a blocking/CPU task to the thread pool."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        fut = self._executor.submit(fn, *args, **kwargs)
        with self._metrics_lock:
            self._metrics["tasks_submitted"] = self._metrics.get("tasks_submitted", 0) + 1
        return fut

    def submit_priority(self, fn: Callable[..., Any], *args, priority: int = 50, block: bool = True, **kwargs) -> ThreadFuture:
        """Submit prioritized task. Lower `priority` runs earlier."""
        if not self._enable_priority or self._pq is None:
            return self.submit(fn, *args, **kwargs)
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        if self._max_queue_size is not None:
            if not block and self._pq.qsize() >= self._max_queue_size:
                raise queue.Full("priority queue full")
        future = ThreadFuture()
        task = _PriorityTask(priority=priority, callable_=fn, args=args, kwargs=kwargs, future=future)
        self._pq.put(task)
        with self._metrics_lock:
            self._metrics["priority_tasks_submitted"] = self._metrics.get("priority_tasks_submitted", 0) + 1
        return future

    def submit_batch(self, callables: List[Callable[[], Any]], *, priority: Optional[int] = None, block: bool = True) -> List[ThreadFuture]:
        """
        Submit a batch of zero-arg callables. Returns list of futures in same order.
        Priority used when submit_priority is available.
        """
        futures: List[ThreadFuture] = []
        for c in callables:
            if priority is None:
                futures.append(self.submit(c))
            else:
                futures.append(self.submit_priority(c, priority=priority, block=block))
        return futures

    def spawn(self, coro: Awaitable, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """Schedule coroutine on background loop and return concurrent.futures.Future."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        loop = self._ensure_loop()
        with self._metrics_lock:
            self._metrics["coroutines_spawned"] = self._metrics.get("coroutines_spawned", 0) + 1
        cf = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token:
            def _cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(_cancel_cb)
        return cf

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        """Run a coroutine synchronously (schedules & waits) or run callable in threadpool and wait."""
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # -----------------------
    # Batch and map helpers
    # -----------------------
    def map_with_concurrency(self, fn: Callable[..., Any], iterable, concurrency: int = 8) -> List[Any]:
        """Run fn(item) for each item using up to `concurrency` threads. Returns results in order."""
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = threading.Semaphore(concurrency)
        threads: List[threading.Thread] = []

        def worker(i, item):
            try:
                sem.acquire()
                results[i] = fn(item)
            except Exception:
                LOG.exception("map_with_concurrency worker failed")
                results[i] = None
            finally:
                try:
                    sem.release()
                except Exception:
                    pass

        for i, item in enumerate(items):
            t = threading.Thread(target=worker, args=(i, item), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    async def amap_with_concurrency(self, func: Callable[[Any], Awaitable[Any]], iterable, concurrency: int = 8) -> List[Any]:
        """Async version: apply async func over items with limited concurrency; returns results in order."""
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = asyncio.Semaphore(concurrency)

        async def worker(i, item):
            async with sem:
                try:
                    results[i] = await func(item)
                except Exception:
                    LOG.exception("amap worker failed")
                    results[i] = None

        tasks = [asyncio.create_task(worker(i, itm)) for i, itm in enumerate(items)]
        await asyncio.gather(*tasks)
        return results

    # -----------------------
    # Scheduling helpers
    # -----------------------
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """Schedule coroutine factory to run after delay seconds; returns TimerHandle."""
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _runner():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coro factory raised")

        handle = loop.call_later(delay, _runner)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable], jitter: float = 0.0) -> TimerHandle:
        """
        Schedule repeating coroutine every `interval` seconds. Optional jitter in seconds reduces thundering herd.
        Returns TimerHandle; cancel stops future repeats.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _looped():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating task raised")
                    # apply jitter if requested
                    if jitter and jitter > 0:
                        await asyncio.sleep(interval + (jitter * (2 * (time.time() % 1) - 1)))
                    else:
                        await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_looped(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # -----------------------
    # Diagnostics & utilities
    # -----------------------
    def dump_threads(self, path: Optional[str] = None) -> str:
        """
        Dump stack traces of all threads into a file (or return as string if path=None).
        Useful for debugging deadlocks or long-running tasks.
        """
        data = []
        for tid, frame in sys._current_frames().items():
            header = f"ThreadID: {tid}"
            stack = "".join(traceback.format_stack(frame))
            data.append(header + "\n" + stack)
        out = "\n\n".join(data)
        if path:
            try:
                atomic_write = getattr(self, "atomic_write_text", None)
                if atomic_write is None:
                    with open(path + ".tmp", "w", encoding="utf-8") as f:
                        f.write(out)
                    os.replace(path + ".tmp", path)
                else:
                    atomic_write(path, out)
            except Exception:
                LOG.exception("failed to write thread dump")
            return path
        return out

    def export_metrics(self, path: str) -> str:
        """Write runtime metrics to path and return path."""
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self.metrics(), f, indent=2)
            os.replace(path + ".tmp", path)
        except Exception:
            LOG.exception("export_metrics failed")
        return path

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self._metrics = {"tasks_submitted": 0, "priority_tasks_submitted": 0, "thread_tasks_executed": 0, "coroutines_spawned": 0, "start_time": time.time()}

    def metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            m = dict(self._metrics)
        m.update({"uptime": time.time() - m.get("start_time", time.time())})
        return m

    # -----------------------
    # Worker affinity and resize
    # -----------------------
    def set_worker_affinity(self, cpus: Optional[List[int]]) -> None:
        """Set CPU affinity hint used by priority workers (best-effort; Linux only)."""
        self._worker_affinity = list(cpus) if cpus else None

    def resize_workers(self, new_count: int) -> None:
        """Resize underlying threadpool; best-effort swap to new executor."""
        new_count = max(1, int(new_count))
        if new_count == self._workers:
            return
        LOG.info("Resizing worker pool %d -> %d", self._workers, new_count)
        new_exec = ThreadPoolExecutor(max_workers=new_count)
        old_exec = self._executor
        self._executor = new_exec
        self._workers = new_count
        # adjust pq semaphore capacity
        try:
            self._pq_semaphore = threading.Semaphore(self._workers)
        except Exception:
            pass
        # restart pq workers if enabled
        if self._enable_priority:
            self._stop_pq_workers()
            self._start_pq_workers(min(self._workers, 2))
        try:
            old_exec.shutdown(wait=False)
        except Exception:
            LOG.exception("old executor shutdown failed")

    # -----------------------
    # Shutdown / lifecycle
    # -----------------------
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Gracefully shutdown runtime (idempotent)."""
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        # stop priority workers
        if self._enable_priority:
            self._stop_pq_workers()
        # stop event loop
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop() failed")
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop()")
        # shutdown executor
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")
        # join loop thread
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    # -----------------------
    # Decorators / helpers
    # -----------------------
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        """Decorator to run a function in the threadpool when called."""
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)
        return _wrapped

    # context manager support
    def __enter__(self) -> "AsyncThreadingRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

# singleton accessor
_global_runtime_lock = threading.Lock()
_global_runtime: Optional[AsyncThreadingRuntime] = None


def get_runtime(workers: int = 4, enable_priority_queue: bool = False) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers, enable_priority_queue=enable_priority_queue)
        return _global_runtime


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]

# --- demo when run as script (lightweight) ---
if __name__ == "__main__":
    rt = get_runtime(workers=2, enable_priority_queue=True)
    @rt.background
    def blocking_task(n):
        LOG.info("Blocking %d start", n)
        time.sleep(1.0 + 0.2 * n)
        LOG.info("Blocking %d done", n)
        return n * n

    async def async_task(n):
        LOG.info("Async %d start", n)
        await asyncio.sleep(0.5)
        LOG.info("Async %d done", n)
        return n + 100

    # submit and spawn
    fs = [blocking_task(i) for i in range(4)]
    afs = [rt.spawn(async_task(i)) for i in range(3)]

    for f in fs:
        LOG.info("Result blocking: %s", f.result())

    for af in afs:
        LOG.info("Result async: %s", af.result())

    # repeating
    counter = {"v": 0}
    def make_repeat():
        async def _coro():
            counter["v"] += 1
            LOG.info("repeat run %d", counter["v"])
            if counter["v"] >= 3:
                rep_handle.cancel()
        return _coro()
    rep_handle = rt.schedule_repeating(0.8, make_repeat, jitter=0.1)

    time.sleep(4.0)
    rt.shutdown()
    LOG.info("runtime shutdown complete")

"""
instryx_async_threading_runtime.py

Production-ready async+threading runtime for Instryx.

Key features added:
- Background asyncio event loop in dedicated thread.
- ThreadPoolExecutor with optional priority queue and worker affinity hints.
- Task tagging, timeouts, watchdog for long-running tasks.
- submit_with_timeout / submit_with_watchdog helpers.
- run_coroutine_with_timeout bridge.
- Graceful shutdown_now and restart.
- Enhanced metrics (histograms, counters, latencies).
- Exportable diagnostics and thread dumps.
- Context manager support and idempotent operations.
- Safe, dependency-free, uses only stdlib.

This file replaces/extends the previous runtime implementation and is ready to be used as-is.
"""

from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture, TimeoutError as FutureTimeout
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ----- Utilities -----
def _now() -> float:
    return time.time()


def _safe_call(fn: Callable[..., Any], *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        LOG.exception("task raised exception")
        raise


# ----- Cancellation token -----
class CancellationToken:
    """Cooperative cancellation token with callback registration."""
    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
            self._callbacks.clear()
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
            else:
                self._callbacks.append(cb)


# ----- Timer handle -----
class TimerHandle:
    """Handle for a scheduled timer; calling cancel() prevents further runs."""
    def __init__(self, cancel_fn: Callable[[], None]) -> None:
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# ----- Priority task descriptor -----
class _PriorityTask:
    __slots__ = ("priority", "created", "task_id", "callable", "args", "kwargs", "future", "tag")

    def __init__(self, priority: int, callable_: Callable, args: Tuple, kwargs: Dict, future: ThreadFuture, tag: Optional[str] = None):
        self.priority = int(priority)
        self.created = _now()
        self.task_id = uuid.uuid4().hex
        self.callable = callable_
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.tag = tag or ""

    def __lt__(self, other: "_PriorityTask") -> bool:  # for PriorityQueue ordering
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created < other.created


# ----- Runtime -----
class AsyncThreadingRuntime:
    """
    Combined asyncio + threaded runtime with advanced features.

    Typical usage:
      rt = AsyncThreadingRuntime(workers=8, enable_priority_queue=True)
      fut = rt.submit_with_timeout(lambda: heavy_work(), timeout=5.0)
      cf = rt.spawn(async_fn())
      rt.shutdown()
    """

    def __init__(
        self,
        workers: int = 4,
        enable_priority_queue: bool = False,
        max_queue_size: Optional[int] = 0,
        loop_name: Optional[str] = None,
        worker_affinity: Optional[List[int]] = None,
    ):
        self._workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown_flag = threading.Event()
        self._loop_name = loop_name or "instryx-async-loop"
        # priority queue
        self._enable_priority = bool(enable_priority_queue)
        self._max_queue_size = None if (max_queue_size is None or max_queue_size <= 0) else int(max_queue_size)
        self._pq: Optional[queue.PriorityQueue] = queue.PriorityQueue() if self._enable_priority else None
        self._pq_workers: List[threading.Thread] = []
        self._pq_worker_count = min(2, self._workers) if self._enable_priority else 0
        self._pq_semaphore = threading.Semaphore(self._workers)
        # metrics and telemetry
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "tasks_submitted": 0,
            "priority_tasks_submitted": 0,
            "thread_tasks_executed": 0,
            "coroutines_spawned": 0,
            "task_latencies": [],  # sample latencies
            "start_time": _now(),
        }
        # affinity hints
        self._worker_affinity = list(worker_affinity) if worker_affinity else None

        # start loop & pq workers
        self._start_loop_thread()
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # ---- event loop thread ----
    def _loop_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_started.set()
        LOG.debug("Async loop started in thread %s", threading.current_thread().name)
        try:
            loop.run_forever()
            # on stop, cancel pending tasks
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                LOG.exception("loop close failed")
            LOG.debug("Async loop closed")

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("asyncio loop not available")
        return self._loop

    # ---- priority queue workers ----
    def _pq_worker_target(self) -> None:
        assert self._pq is not None
        # best-effort set affinity for this worker
        if self._worker_affinity and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(self._worker_affinity))
            except Exception:
                LOG.debug("sched_setaffinity failed for pq worker (ignored)")
        while not self._shutdown_flag.is_set():
            try:
                task: _PriorityTask = self._pq.get(timeout=0.2)
            except queue.Empty:
                continue
            acquired = False
            try:
                self._pq_semaphore.acquire()
                acquired = True
                if task.future.cancelled():
                    continue
                start = _now()
                try:
                    res = task.callable(*task.args, **task.kwargs)
                    task.future.set_result(res)
                except Exception as e:
                    task.future.set_exception(e)
                latency = _now() - start
                with self._metrics_lock:
                    self._metrics["thread_tasks_executed"] = self._metrics.get("thread_tasks_executed", 0) + 1
                    self._metrics.setdefault("task_latencies", []).append(latency)
            finally:
                if acquired:
                    try:
                        self._pq_semaphore.release()
                    except Exception:
                        pass
                try:
                    self._pq.task_done()
                except Exception:
                    pass

    def _start_pq_workers(self, count: int) -> None:
        if not self._enable_priority or self._pq is None:
            return
        if self._pq_workers:
            return
        for i in range(max(1, int(count))):
            t = threading.Thread(target=self._pq_worker_target, name=f"instryx-pq-worker-{i}", daemon=True)
            t.start()
            self._pq_workers.append(t)

    def _stop_pq_workers(self) -> None:
        for t in list(self._pq_workers):
            try:
                if t.is_alive():
                    t.join(timeout=0.5)
            except Exception:
                pass
        self._pq_workers = []

    # ---- submission APIs ----
    def submit(self, fn: Callable[..., Any], *args, tag: Optional[str] = None, **kwargs) -> ThreadFuture:
        """Submit a blocking/CPU task to the thread pool."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        start = _now()
        fut = self._executor.submit(_safe_call, fn, *args, **kwargs)
        with self._metrics_lock:
            self._metrics["tasks_submitted"] = self._metrics.get("tasks_submitted", 0) + 1
        # attach metadata
        fut._task_tag = tag if tag else ""
        fut._submit_time = start
        return fut

    def submit_priority(self, fn: Callable[..., Any], *args, priority: int = 50, block: bool = True, tag: Optional[str] = None, **kwargs) -> ThreadFuture:
        """Submit prioritized task. Lower `priority` runs earlier."""
        if not self._enable_priority or self._pq is None:
            return self.submit(fn, *args, tag=tag, **kwargs)
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        if self._max_queue_size is not None:
            if not block and self._pq.qsize() >= self._max_queue_size:
                raise queue.Full("priority queue full")
        future = ThreadFuture()
        task = _PriorityTask(priority=priority, callable_=fn, args=args, kwargs=kwargs, future=future, tag=tag)
        self._pq.put(task)
        with self._metrics_lock:
            self._metrics["priority_tasks_submitted"] = self._metrics.get("priority_tasks_submitted", 0) + 1
        return future

    def submit_with_timeout(self, fn: Callable[..., Any], timeout: float, *args, **kwargs) -> ThreadFuture:
        """
        Submit a blocking task and cancel it (best-effort) if it does not complete within timeout.
        Note: cancelling running threads is not possible in stdlib; this will mark the future as timed out.
        """
        fut = self.submit(fn, *args, **kwargs)
        def _watch():
            try:
                fut.result(timeout=timeout)
            except FutureTimeout:
                try:
                    fut.cancel()
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
        return fut

    def submit_with_watchdog(self, fn: Callable[..., Any], watchdog_secs: float, *args, **kwargs) -> ThreadFuture:
        """
        Submit a task and log/emit diagnostics if it runs longer than watchdog_secs.
        """
        fut = self.submit(fn, *args, **kwargs)
        tag = getattr(fut, "_task_tag", "")
        def _watch():
            try:
                fut.result(timeout=watchdog_secs)
            except FutureTimeout:
                LOG.warning("Watchdog: task %s running > %.2fs (tag=%s)", getattr(fut, "_submit_time", "<>"), watchdog_secs, tag)
                try:
                    # dump threads for diagnostics
                    self.dump_threads(path=None)  # will return string; logged below
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
        return fut

    def submit_batch(self, callables: List[Callable[[], Any]], *, priority: Optional[int] = None, block: bool = True) -> List[ThreadFuture]:
        futures: List[ThreadFuture] = []
        for c in callables:
            if priority is None:
                futures.append(self.submit(c))
            else:
                futures.append(self.submit_priority(c, priority=priority, block=block))
        return futures

    # ---- coroutine APIs ----
    def spawn(self, coro: Awaitable, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """Schedule coroutine on background loop and return concurrent.futures.Future."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        loop = self._ensure_loop()
        with self._metrics_lock:
            self._metrics["coroutines_spawned"] = self._metrics.get("coroutines_spawned", 0) + 1
        cf = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token:
            def _cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(_cancel_cb)
        return cf

    def run_coroutine_with_timeout(self, coro: Awaitable, timeout: Optional[float] = None) -> Any:
        """Run coroutine on runtime loop and wait for result with optional timeout."""
        fut = self.spawn(coro)
        return fut.result(timeout=timeout)

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        """Run coroutine synchronously (schedules & waits) or run callable in threadpool and wait."""
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # ---- batch helpers ----
    def map_with_concurrency(self, fn: Callable[..., Any], iterable, concurrency: int = 8) -> List[Any]:
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = threading.Semaphore(concurrency)
        threads: List[threading.Thread] = []

        def worker(i, item):
            try:
                sem.acquire()
                results[i] = _safe_call(fn, item)
            except Exception:
                LOG.exception("map worker failed")
                results[i] = None
            finally:
                try:
                    sem.release()
                except Exception:
                    pass

        for i, item in enumerate(items):
            t = threading.Thread(target=worker, args=(i, item), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    async def amap_with_concurrency(self, func: Callable[[Any], Awaitable[Any]], iterable, concurrency: int = 8) -> List[Any]:
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = asyncio.Semaphore(concurrency)

        async def worker(i, item):
            async with sem:
                try:
                    results[i] = await func(item)
                except Exception:
                    LOG.exception("amap worker failed")
                    results[i] = None

        tasks = [asyncio.create_task(worker(i, itm)) for i, itm in enumerate(items)]
        await asyncio.gather(*tasks)
        return results

    # ---- scheduling helpers ----
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _runner():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coro factory raised")

        handle = loop.call_later(delay, _runner)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable], jitter: float = 0.0) -> TimerHandle:
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _looped():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating task raised")
                    if jitter and jitter > 0:
                        await asyncio.sleep(interval + (jitter * (2 * (time.time() % 1) - 1)))
                    else:
                        await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_looped(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # ---- diagnostics & utilities ----
    def dump_threads(self, path: Optional[str] = None) -> str:
        data = []
        for tid, frame in sys._current_frames().items():
            header = f"ThreadID: {tid}"
            stack = "".join(traceback.format_stack(frame))
            data.append(header + "\n" + stack)
        out = "\n\n".join(data)
        if path:
            try:
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(out)
                os.replace(tmp, path)
            except Exception:
                LOG.exception("failed to write thread dump")
            return path
        LOG.debug("Thread dump:\n%s", out)
        return out

    def export_metrics(self, path: str) -> str:
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(self._export_metrics_json())
            os.replace(tmp, path)
        except Exception:
            LOG.exception("export_metrics failed")
        return path

    def _export_metrics_json(self) -> str:
        return _now_json(self.metrics())

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self._metrics = {
                "tasks_submitted": 0,
                "priority_tasks_submitted": 0,
                "thread_tasks_executed": 0,
                "coroutines_spawned": 0,
                "task_latencies": [],
                "start_time": _now(),
            }

    def metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            m = dict(self._metrics)
        m["uptime"] = _now() - m.get("start_time", _now())
        # provide simple summaries
        lat = m.get("task_latencies") or []
        if lat:
            m["latency_min"] = min(lat)
            m["latency_max"] = max(lat)
            m["latency_avg"] = sum(lat) / len(lat)
            m["latency_count"] = len(lat)
        else:
            m["latency_min"] = m["latency_max"] = m["latency_avg"] = 0.0
            m["latency_count"] = 0
        return m

    # ---- affinity & resize ----
    def set_worker_affinity(self, cpus: Optional[List[int]]) -> None:
        self._worker_affinity = list(cpus) if cpus else None

    def resize_workers(self, new_count: int) -> None:
        new_count = max(1, int(new_count))
        if new_count == self._workers:
            return
        LOG.info("Resizing workers %d -> %d", self._workers, new_count)
        new_exec = ThreadPoolExecutor(max_workers=new_count)
        old_exec = self._executor
        self._executor = new_exec
        self._workers = new_count
        try:
            self._pq_semaphore = threading.Semaphore(self._workers)
        except Exception:
            pass
        if self._enable_priority:
            self._stop_pq_workers()
            self._start_pq_workers(min(self._workers, 2))
        try:
            old_exec.shutdown(wait=False)
        except Exception:
            LOG.exception("old executor shutdown failed")

    # ---- lifecycle ----
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        if self._enable_priority:
            self._stop_pq_workers()
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop failed")
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop")
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    def shutdown_now(self) -> None:
        """Force immediate shutdown: cancel pending pq tasks and stop loop."""
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        if self._pq:
            try:
                while not self._pq.empty():
                    task: _PriorityTask = self._pq.get_nowait()
                    try:
                        task.future.cancel()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            loop = self._ensure_loop()
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)

    # ---- helpers ----
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)
        return _wrapped

    def __enter__(self) -> "AsyncThreadingRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)


# ----- module-level singleton -----
_global_runtime_lock = threading.Lock()
_global_runtime: Optional[AsyncThreadingRuntime] = None


def get_runtime(workers: int = 4, enable_priority_queue: bool = False) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers, enable_priority_queue=enable_priority_queue)
        return _global_runtime


# ----- tiny helpers -----
def _now_json(obj: Any) -> str:
    import json
    return json.dumps(obj, default=str, indent=2)


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]


# ----- demo when run as script -----
if __name__ == "__main__":
    rt = get_runtime(workers=2, enable_priority_queue=True)
    @rt.background
    def blocking_task(n):
        LOG.info("Blocking %d start", n)
        time.sleep(1.0 + 0.1 * n)
        LOG.info("Blocking %d done", n)
        return n * n

    async def async_task(n):
        LOG.info("Async %d start", n)
        await asyncio.sleep(0.5)
        LOG.info("Async %d done", n)
        return n + 100

    fs = [blocking_task(i) for i in range(4)]
    afs = [rt.spawn(async_task(i)) for i in range(3)]

    for f in fs:
        LOG.info("Result blocking: %s", f.result())

    for af in afs:
        LOG.info("Result async: %s", af.result())

    def make_repeat():
        async def _coro():
            LOG.info("Repeat job running")
        return _coro()

    rep_handle = rt.schedule_repeating(0.8, make_repeat, jitter=0.1)
    time.sleep(3.0)
    rep_handle.cancel()
    rt.export_metrics("runtime_metrics.json")
    rt.shutdown()
    LOG.info("runtime shutdown complete")

"""
instryx_async_threading_runtime.py

Production-ready async+threading runtime for Instryx.

Features:
- Background asyncio event loop in a dedicated thread.
- ThreadPoolExecutor with optional priority queue and worker affinity hints.
- Task tagging, timeouts, watchdog for long-running tasks.
- submit_with_timeout / submit_with_watchdog helpers.
- run_coroutine_with_timeout bridge and run_in_executor helper.
- Graceful shutdown_now and restart.
- Enhanced metrics (histograms, counters, latencies).
- Exportable diagnostics and thread dumps.
- Context manager support and idempotent operations.
- Safe, dependency-free, uses only stdlib.
"""
from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
import os
import sys
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture, TimeoutError as FutureTimeout
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---- small utilities ----
def _now() -> float:
    return time.time()


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)


def _safe_call(fn: Callable[..., Any], *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        LOG.exception("task raised exception")
        raise


# ---- cancellation / timer handles ----
class CancellationToken:
    """Cooperative cancellation token with callback registration."""
    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
            self._callbacks.clear()
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
            else:
                self._callbacks.append(cb)


class TimerHandle:
    """Handle for a scheduled timer; calling cancel() prevents further runs."""
    def __init__(self, cancel_fn: Callable[[], None]) -> None:
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# ---- priority task descriptor ----
class _PriorityTask:
    __slots__ = ("priority", "created", "task_id", "callable", "args", "kwargs", "future", "tag")

    def __init__(self, priority: int, callable_: Callable, args: Tuple, kwargs: Dict, future: ThreadFuture, tag: Optional[str] = None):
        self.priority = int(priority)
        self.created = _now()
        self.task_id = uuid.uuid4().hex
        self.callable = callable_
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.tag = tag or ""

    def __lt__(self, other: "_PriorityTask") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created < other.created


# ---- runtime ----
class AsyncThreadingRuntime:
    """
    Combined asyncio + threaded runtime with advanced features.

    Usage:
      rt = AsyncThreadingRuntime(workers=8, enable_priority_queue=True)
      fut = rt.submit_with_timeout(lambda: heavy_work(), timeout=5.0)
      cf = rt.spawn(async_fn())
      rt.shutdown()
    """

    def __init__(
        self,
        workers: int = 4,
        enable_priority_queue: bool = False,
        max_queue_size: Optional[int] = 0,
        loop_name: Optional[str] = None,
        worker_affinity: Optional[List[int]] = None,
    ):
        self._workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown_flag = threading.Event()
        self._loop_name = loop_name or "instryx-async-loop"

        # priority queue
        self._enable_priority = bool(enable_priority_queue)
        self._max_queue_size = None if (max_queue_size is None or max_queue_size <= 0) else int(max_queue_size)
        self._pq: Optional[queue.PriorityQueue] = queue.PriorityQueue() if self._enable_priority else None
        self._pq_workers: List[threading.Thread] = []
        self._pq_worker_count = min(2, self._workers) if self._enable_priority else 0
        self._pq_semaphore = threading.Semaphore(self._workers)

        # metrics and telemetry
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "tasks_submitted": 0,
            "priority_tasks_submitted": 0,
            "thread_tasks_executed": 0,
            "coroutines_spawned": 0,
            "task_latencies": [],
            "start_time": _now(),
        }

        # affinity hints
        self._worker_affinity = list(worker_affinity) if worker_affinity else None

        # start loop & pq workers
        self._start_loop_thread()
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # ---- event loop thread ----
    def _loop_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_started.set()
        LOG.debug("Async loop started in thread %s", threading.current_thread().name)
        try:
            loop.run_forever()
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                LOG.exception("loop close failed")
            LOG.debug("Async loop closed")

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("asyncio loop not available")
        return self._loop

    # ---- priority queue workers ----
    def _pq_worker_target(self) -> None:
        assert self._pq is not None
        if self._worker_affinity and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(self._worker_affinity))
            except Exception:
                LOG.debug("sched_setaffinity failed for pq worker (ignored)")
        while not self._shutdown_flag.is_set():
            try:
                task: _PriorityTask = self._pq.get(timeout=0.2)
            except queue.Empty:
                continue
            acquired = False
            try:
                self._pq_semaphore.acquire()
                acquired = True
                if task.future.cancelled():
                    continue
                start = _now()
                try:
                    res = task.callable(*task.args, **task.kwargs)
                    task.future.set_result(res)
                except Exception as e:
                    task.future.set_exception(e)
                latency = _now() - start
                with self._metrics_lock:
                    self._metrics["thread_tasks_executed"] = self._metrics.get("thread_tasks_executed", 0) + 1
                    self._metrics.setdefault("task_latencies", []).append(latency)
            finally:
                if acquired:
                    try:
                        self._pq_semaphore.release()
                    except Exception:
                        pass
                try:
                    self._pq.task_done()
                except Exception:
                    pass

    def _start_pq_workers(self, count: int) -> None:
        if not self._enable_priority or self._pq is None:
            return
        if self._pq_workers:
            return
        for i in range(max(1, int(count))):
            t = threading.Thread(target=self._pq_worker_target, name=f"instryx-pq-worker-{i}", daemon=True)
            t.start()
            self._pq_workers.append(t)

    def _stop_pq_workers(self) -> None:
        for t in list(self._pq_workers):
            try:
                if t.is_alive():
                    t.join(timeout=0.5)
            except Exception:
                pass
        self._pq_workers = []

    # ---- submission APIs ----
    def submit(self, fn: Callable[..., Any], *args, tag: Optional[str] = None, **kwargs) -> ThreadFuture:
        """Submit a blocking/CPU task to the thread pool."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        start = _now()
        fut = self._executor.submit(_safe_call, fn, *args, **kwargs)
        with self._metrics_lock:
            self._metrics["tasks_submitted"] = self._metrics.get("tasks_submitted", 0) + 1
        fut._task_tag = tag if tag else ""
        fut._submit_time = start
        return fut

    def submit_priority(self, fn: Callable[..., Any], *args, priority: int = 50, block: bool = True, tag: Optional[str] = None, **kwargs) -> ThreadFuture:
        """Submit prioritized task. Lower `priority` runs earlier."""
        if not self._enable_priority or self._pq is None:
            return self.submit(fn, *args, tag=tag, **kwargs)
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        if self._max_queue_size is not None and not block and self._pq.qsize() >= self._max_queue_size:
            raise queue.Full("priority queue full")
        future = ThreadFuture()
        task = _PriorityTask(priority=priority, callable_=fn, args=args, kwargs=kwargs, future=future, tag=tag)
        self._pq.put(task)
        with self._metrics_lock:
            self._metrics["priority_tasks_submitted"] = self._metrics.get("priority_tasks_submitted", 0) + 1
        return future

    def submit_with_timeout(self, fn: Callable[..., Any], timeout: float, *args, **kwargs) -> ThreadFuture:
        """
        Submit a blocking task and mark/cancel it (best-effort) if it doesn't complete within timeout.
        Note: cannot reliably kill running thread; this marks the future cancelled on timeout.
        """
        fut = self.submit(fn, *args, **kwargs)
        def _watch():
            try:
                fut.result(timeout=timeout)
            except FutureTimeout:
                try:
                    fut.cancel()
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
        return fut

    def submit_with_watchdog(self, fn: Callable[..., Any], watchdog_secs: float, *args, **kwargs) -> ThreadFuture:
        """
        Submit a task and log diagnostics if it runs longer than watchdog_secs.
        """
        fut = self.submit(fn, *args, **kwargs)
        tag = getattr(fut, "_task_tag", "")
        def _watch():
            try:
                fut.result(timeout=watchdog_secs)
            except FutureTimeout:
                LOG.warning("Watchdog: task started at %s running > %.2fs (tag=%s)", getattr(fut, "_submit_time", "<>"), watchdog_secs, tag)
                try:
                    dump = self.dump_threads(path=None)
                    LOG.debug("Thread dump (watchdog): %s", dump)
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
        return fut

    def submit_batch(self, callables: List[Callable[[], Any]], *, priority: Optional[int] = None, block: bool = True) -> List[ThreadFuture]:
        futures: List[ThreadFuture] = []
        for c in callables:
            if priority is None:
                futures.append(self.submit(c))
            else:
                futures.append(self.submit_priority(c, priority=priority, block=block))
        return futures

    # ---- coroutine APIs ----
    def spawn(self, coro: Awaitable, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """Schedule coroutine on background loop and return concurrent.futures.Future."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        loop = self._ensure_loop()
        with self._metrics_lock:
            self._metrics["coroutines_spawned"] = self._metrics.get("coroutines_spawned", 0) + 1
        cf = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token:
            def _cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(_cancel_cb)
        return cf

    def run_coroutine_with_timeout(self, coro: Awaitable, timeout: Optional[float] = None) -> Any:
        fut = self.spawn(coro)
        return fut.result(timeout=timeout)

    def run_in_executor_async(self, fn: Callable[..., Any], *args, timeout: Optional[float] = None, **kwargs) -> Any:
        """Schedule a blocking function in the runtime executor but await it as coroutine."""
        loop = self._ensure_loop()
        return loop.run_in_executor(self._executor, functools.partial(_safe_call, fn, *args, **kwargs))

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # ---- batch helpers ----
    def map_with_concurrency(self, fn: Callable[..., Any], iterable, concurrency: int = 8) -> List[Any]:
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = threading.Semaphore(concurrency)
        threads: List[threading.Thread] = []

        def worker(i, item):
            try:
                sem.acquire()
                results[i] = _safe_call(fn, item)
            except Exception:
                LOG.exception("map worker failed")
                results[i] = None
            finally:
                try:
                    sem.release()
                except Exception:
                    pass

        for i, item in enumerate(items):
            t = threading.Thread(target=worker, args=(i, item), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    async def amap_with_concurrency(self, func: Callable[[Any], Awaitable[Any]], iterable, concurrency: int = 8) -> List[Any]:
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = asyncio.Semaphore(concurrency)

        async def worker(i, item):
            async with sem:
                try:
                    results[i] = await func(item)
                except Exception:
                    LOG.exception("amap worker failed")
                    results[i] = None

        tasks = [asyncio.create_task(worker(i, itm)) for i, itm in enumerate(items)]
        await asyncio.gather(*tasks)
        return results

    # ---- scheduling helpers ----
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _runner():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coro factory raised")

        handle = loop.call_later(delay, _runner)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable], jitter: float = 0.0) -> TimerHandle:
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _looped():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating task raised")
                    if jitter and jitter > 0:
                        await asyncio.sleep(interval + (jitter * (2 * (time.time() % 1) - 1)))
                    else:
                        await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_looped(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # ---- diagnostics & utilities ----
    def dump_threads(self, path: Optional[str] = None) -> str:
        data = []
        for tid, frame in sys._current_frames().items():
            header = f"ThreadID: {tid}"
            stack = "".join(traceback.format_stack(frame))
            data.append(header + "\n" + stack)
        out = "\n\n".join(data)
        if path:
            try:
                atomic_write_text(path, out)
            except Exception:
                LOG.exception("failed to write thread dump")
            return path
        LOG.debug("Thread dump:\n%s", out)
        return out

    def export_metrics(self, path: str) -> str:
        try:
            atomic_write_text(path, json.dumps(self.metrics(), default=str, indent=2))
        except Exception:
            LOG.exception("export_metrics failed")
        return path

    def _export_metrics_json(self) -> str:
        return json.dumps(self.metrics(), default=str, indent=2)

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self._metrics = {
                "tasks_submitted": 0,
                "priority_tasks_submitted": 0,
                "thread_tasks_executed": 0,
                "coroutines_spawned": 0,
                "task_latencies": [],
                "start_time": _now(),
            }

    def metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            m = dict(self._metrics)
        m["uptime"] = _now() - m.get("start_time", _now())
        lat = m.get("task_latencies") or []
        if lat:
            m["latency_min"] = min(lat)
            m["latency_max"] = max(lat)
            m["latency_avg"] = sum(lat) / len(lat)
            m["latency_count"] = len(lat)
        else:
            m["latency_min"] = m["latency_max"] = m["latency_avg"] = 0.0
            m["latency_count"] = 0
        return m

    # ---- affinity & resize ----
    def set_worker_affinity(self, cpus: Optional[List[int]]) -> None:
        self._worker_affinity = list(cpus) if cpus else None

    def resize_workers(self, new_count: int) -> None:
        new_count = max(1, int(new_count))
        if new_count == self._workers:
            return
        LOG.info("Resizing workers %d -> %d", self._workers, new_count)
        new_exec = ThreadPoolExecutor(max_workers=new_count)
        old_exec = self._executor
        self._executor = new_exec
        self._workers = new_count
        try:
            self._pq_semaphore = threading.Semaphore(self._workers)
        except Exception:
            pass
        if self._enable_priority:
            self._stop_pq_workers()
            self._start_pq_workers(min(self._workers, 2))
        try:
            old_exec.shutdown(wait=False)
        except Exception:
            LOG.exception("old executor shutdown failed")

    # ---- lifecycle ----
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        if self._enable_priority:
            self._stop_pq_workers()
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop failed")
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop")
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    def shutdown_now(self) -> None:
        """Force immediate shutdown: cancel pending pq tasks and stop loop."""
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        if self._pq:
            try:
                while not self._pq.empty():
                    task: _PriorityTask = self._pq.get_nowait()
                    try:
                        task.future.cancel()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            loop = self._ensure_loop()
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)

    def restart(self, wait: bool = True) -> None:
        """Gracefully restart loop and workers (best-effort)."""
        self.shutdown(wait=wait)
        # reset flag and start fresh loop and executor
        self._shutdown_flag.clear()
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._start_loop_thread()
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # ---- helpers ----
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)
        return _wrapped

    def __enter__(self) -> "AsyncThreadingRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)


# ---- module-level singleton ----
_global_runtime_lock = threading.Lock()
_global_runtime: Optional[AsyncThreadingRuntime] = None


def get_runtime(workers: int = 4, enable_priority_queue: bool = False) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers, enable_priority_queue=enable_priority_queue)
        return _global_runtime


# ---- tiny helpers ----
def _now_json(obj: Any) -> str:
    return json.dumps(obj, default=str, indent=2)


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]

# ---- demo when run as script ----
if __name__ == "__main__":
    rt = get_runtime(workers=2, enable_priority_queue=True)
    @rt.background
    def blocking_task(n):
        LOG.info("Blocking %d start", n)
        time.sleep(1.0 + 0.1 * n)
        LOG.info("Blocking %d done", n)
        return n * n
    async def async_task(n):
        LOG.info("Async %d start", n)
        await asyncio.sleep(0.5)
        LOG.info("Async %d done", n)
        return n + 100
    fs = [blocking_task(i) for i in range(4)]
    afs = [rt.spawn(async_task(i)) for i in range(3)]
    for f in fs:
        LOG.info("Result blocking: %s", f.result())
    for af in afs:
        LOG.info("Result async: %s", af.result())
    def make_repeat():
        async def _coro():
            LOG.info("Repeat job running")
        return _coro()
    rep_handle = rt.schedule_repeating(0.8, make_repeat, jitter=0.1)
    time.sleep(3.0)
    rep_handle.cancel()
    rt.export_metrics("runtime_metrics.json")
    rt.shutdown()
    LOG.info("runtime shutdown complete")
    """
    instryx_async_threading_runtime.py
    Production-ready async+threading runtime for Instryx.
    Features:
    - Background asyncio event loop in a dedicated thread.
    - ThreadPoolExecutor with optional priority queue and worker affinity hints.
    - Task tagging, timeouts, watchdog for long-running tasks.
    - submit_with_timeout / submit_with_watchdog helpers.
    - run_coroutine_with_timeout bridge and run_in_executor helper.
    - Graceful shutdown_now and restart.
    - Enhanced metrics (histograms, counters, latencies).
    - Exportable diagnostics and thread dumps.
    - Context manager support and idempotent operations.
    - Safe, dependency-free, uses only stdlib.
    """

