"""One thread runs every inference, fed by a bounded queue (docs/spec.md, 3.3).

CTranslate2 and onnxruntime calls block, and several speakers can finish an utterance at once; they
queue here instead of contending for the GPU or blocking the event loop.
"""

import asyncio
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any


class QueueFull(Exception):
    pass


class GpuWorker:
    def __init__(self, depth: int = 32, name: str = "gpu-worker", label: str = "transcription"):
        self._label = label
        self._queue: queue.Queue[tuple[Callable[[], Any], Future] | None] = queue.Queue(maxsize=depth)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while (job := self._queue.get()) is not None:
            fn, future = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as error:  # handed to the caller, never kills the worker
                future.set_exception(error)

    def submit(self, fn: Callable[[], Any]) -> Future:
        future: Future = Future()
        try:
            self._queue.put_nowait((fn, future))
        except queue.Full:
            raise QueueFull(f"{self._label} queue is full") from None
        return future

    async def run(self, fn: Callable[[], Any]) -> Any:
        return await asyncio.wrap_future(self.submit(fn))

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)
